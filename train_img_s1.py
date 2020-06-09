from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from reid import data_manager
from reid.dataset_loader import ImageDataset, VideoDataset, VideoDataset_time
from reid import transforms as T
from reid import models
from reid.losses import CrossEntropyLabelSmooth, TripletLoss, DeepSupervision
from reid.utils.iotools import save_checkpoint, check_isfile
from reid.utils.avgmeter import AverageMeter
from reid.utils.logger import Logger
from reid.utils.torchtools import count_num_param
from reid.utils.reidtools import visualize_ranked_results
from reid.eval_metrics import evaluate, eval_dukemtmcreid
from reid.samplers import RandomIdentitySampler, ClassIdentitySampler
from reid.optimizers import init_optim

parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('--root', type=str, default='/gdata1/xieqk/reid-dataset',
                    help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid-tracklet',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=8, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--seq-len', type=int, default=15,
                    help="number of images to sample in a tracklet")
# Optimization options
parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=50, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=256, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=128, type=int,
                    help="test batch size (number of tracklets)")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[10, 20, 30, 40], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=4.0,
                    help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])
# Miscs
parser.add_argument('--print-freq', type=int, default=10,
                    help="print frequency")
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--load-weights', type=str, default='',
                    help="load pretrained weights but ignores layers that don't match in size")
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--eval-step', type=int, default=10,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='')
parser.add_argument('--use-cpu', action='store_true',
                    help="use cpu")
parser.add_argument('--gpu-devices', '-g', default='0,1', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--use-avai-gpus', action='store_true',
                    help="use available gpus instead of specified devices (this is useful when using managed clusters)")
parser.add_argument('--visualize-ranks', action='store_true',
                    help="visualize ranked results, only available in evaluation mode (default: False)")

# global variables
args = parser.parse_args()
best_rank1 = -np.inf

def main():
    global args, best_rank1

    torch.set_num_threads(args.workers)

    torch.manual_seed(args.seed)
    if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = osp.join('logs', '%s_L%.4f_E%d_B%d_S%s_M%.1f_N%d_G%.1f_%s'%(args.dataset, args.lr, args.max_epoch, args.train_batch, str(args.stepsize), args.margin, args.num_instances, args.gamma, time.strftime("%Y%m%d-%H%M%S", time.localtime())))

    if not args.evaluate:
        writer = SummaryWriter(log_dir=save_dir)
        sys.stdout = Logger(osp.join(save_dir, 'log_train_%s_L%.4f_E%d_B%d_S%s_M%.1f_N%d_G%.1f_%s.txt'%(args.dataset, args.lr, args.max_epoch, args.train_batch, str(args.stepsize), args.margin, args.num_instances, args.gamma, time.strftime("%Y%m%d-%H%M%S", time.localtime()))))
    else:
        sys.stdout = Logger(osp.join(save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_imgreid_dataset(root=args.root, name=args.dataset)

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False


    # decompose tracklets into images for image-based training
    train_data = dataset.get_train_tracklets()
    new_train = []
    for img_paths, pid, camid, _, _, _ in train_data:
        for img_path in img_paths:
            new_train.append((img_path, pid, camid))

    trainloader = DataLoader(
        ImageDataset(new_train, transform=transform_train),
        # sampler=RandomIdentitySampler(new_train, args.train_batch, args.num_instances),
        sampler=ClassIdentitySampler(new_train, args.train_batch, args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        # VideoDataset(dataset.query, seq_len=args.seq_len, sample='evenly', transform=transform_test),
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, loss={'htri'})
    print("Model size: {:.3f} M".format(count_num_param(model)))

    criterion_htri = TripletLoss(margin=args.margin)

    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)

    if args.load_weights and check_isfile(args.load_weights):
        # load pretrained weights but ignore layers that don't match in size
        checkpoint = torch.load(args.load_weights)
        pretrain_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.resume and check_isfile(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_rank1 = checkpoint['rank1']
        print("Loaded checkpoint from '{}'".format(args.resume))
        print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, best_rank1))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        distmat = time_test(model, queryloader, galleryloader, args.pool, use_gpu, return_distmat=True)
        print(len(distmat), len(distmat[0]))
        # if args.visualize_ranks:
        #     visualize_ranked_results(
        #         distmat, dataset,
        #         save_dir=osp.join(save_dir, 'ranked_results'),
        #         topk=20,
        #     )
        return

    start_time = time.time()
    train_time = 0
    best_epoch = args.start_epoch
    print("==> Start training")

    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        # test before train
        if epoch == 0:
            mAP, rank1, rank5, rank10, rank20 = test(model, queryloader, galleryloader, use_gpu)
            res_dict = {
                'mAP': mAP,
                'rank-1': rank1,
                'rank-5': rank5,
                'rank-10': rank10,
                'rank-20': rank20,
            }

            writer.add_scalars('scalar/precision', res_dict, epoch)

        train(epoch, model, criterion_htri, optimizer, trainloader, use_gpu, writer)
        train_time += round(time.time() - start_train_time)

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            mAP, rank1, rank5, rank10, rank20 = test(model, queryloader, galleryloader, use_gpu)
            res_dict = {
                'mAP': mAP,
                'rank-1': rank1,
                'rank-5': rank5,
                'rank-10': rank10,
                'rank-20': rank20,
            }

            writer.add_scalars('scalar/precision', res_dict, epoch+1)

            is_best = rank1 > best_rank1

            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            if is_best:
                # save_checkpoint({
                #     'state_dict': state_dict,
                #     'rank1': rank1,
                #     'epoch': epoch,
                # }, is_best, osp.join(save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))
                save_checkpoint({
                    'state_dict': state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                }, fpath=osp.join(save_dir, 's1_best_model' + '.pth.tar'))

        train_data = dataset.get_train_tracklets()
        new_train = []
        for img_paths, pid, camid, _, _, _ in train_data:
            for img_path in img_paths:
                new_train.append((img_path, pid, camid))

        trainloader = DataLoader(
            ImageDataset(new_train, transform=transform_train),
            # sampler=RandomIdentitySampler(new_train, args.train_batch, args.num_instances),
            sampler=ClassIdentitySampler(new_train, args.train_batch, args.num_instances),
            batch_size=args.train_batch, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True,
        )

    save_checkpoint({
        'state_dict': state_dict,
        'rank1': rank1,
        'epoch': epoch,
    }, False, osp.join(save_dir, 's1_checkpoint_final' + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, criterion_htri, optimizer, trainloader, use_gpu, writer):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        features = model(imgs)
        if isinstance(features, (tuple, list)):
            loss = DeepSupervision(criterion_htri, features, pids)
        else:
            loss = criterion_htri(features, pids)

        writer.add_scalar('scalar/loss', loss.item(), epoch*len(trainloader) + batch_idx + 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), pids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

        end = time.time()


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=False)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    if return_distmat:
        print(cmc)
        # np.save('cmc-duke-33.npy', cmc)
        return distmat
    return mAP, cmc[1-1], cmc[5-1], cmc[10-1], cmc[20-1]

if __name__ == '__main__':
    main()
