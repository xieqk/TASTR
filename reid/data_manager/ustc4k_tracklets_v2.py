from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import copy
import re
import pickle
import random
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
import shutil
# from scipy.misc import imsave
import matplotlib.pyplot as plt

import torch

from reid.utils.iotools import mkdir_if_missing, write_json, read_json
from reid.utils.sp_utils import *

def unpkl(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def hist_plot(data, save_path):
    plt.figure(figsize=(15, 10))
    n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.xlabel('Dist')
    plt.ylabel('Cnt')
    plt.savefig(save_path)
    plt.close


class USTC4k_tracklets_v2(object):
    """
    USTC4kVidReID

    Dataset statistics:
    # identities:
    # tracklets:
    """
    # cam = []
    # for i in range(6):
    #     data = unpkl('cam%d_filtered.pkl'%(i+1))
    #     print(len(data.keys()), 'haha')
    #     cam.append(data)

    def __init__(self, root, min_seq_len=0, verbose=True, **kwargs):
        self.cam = []
        for i in range(6):
            data = unpkl(osp.join(root, 'campus4k/tracklet', 'cam%d_filtered.pkl'%(i+1)))
            print(i, ':', len(data.keys()))
            self.cam.append(data)
        self.min_seq_len = min_seq_len
        self.curr_train_cam = 0
        self.sample_num = 100
        self.sample_num_s2 = 50
        self.num_per_person = 60
        self.num_per_person_s2 = 60
        self.min_seq_len_tracklet = 60
        assert self.num_per_person <= self.min_seq_len_tracklet
        self.tracklets_dir = osp.join(root, 'campus4k/tracklet')
        self.dataset_dir = osp.join(root, 'campus4k')

        self.split_query_json_path = osp.join(self.dataset_dir, 'split_query.json')
        self.split_gallery_json_path = osp.join(self.dataset_dir, 'split_gallery.json')

        self._check_before_run()
        self._relabel_cam()
        self._get_train_tids()
        # self._sample()
        self._process_dir_tracklet()
        # self._process_dir_dataset()

        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        # # train, num_train_tracklets, num_train_pids, num_imgs_train = \
        # #     self._process_dir(self.train_dir, self.split_train_json_path, relabel=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
            self._process_dir(self.query_dir, self.split_query_json_path, relabel=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
            self._process_dir(self.gallery_dir, self.split_gallery_json_path, relabel=False)

        # num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        # train_num = np.sum(num_imgs_train)
        # query_num = np.sum(num_imgs_query)
        # gallery_num = np.sum(num_imgs_gallery)
        # num_total_imgs = train_num + query_num + gallery_num
        # min_num = np.min(num_imgs_per_tracklet)
        # max_num = np.max(num_imgs_per_tracklet)
        # avg_num = np.mean(num_imgs_per_tracklet)

        # num_total_pids = num_train_pids + num_query_pids
        # num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        # print(num_query_pids, num_query_tracklets, query_num)

        # if verbose:
        #     print("=> USTC4k_181213 loaded")
        #     print("Dataset statistics:")
        #     print("  ------------------------------")
        #     print("  subset   | # ids | # tracklets |  # imgs  |")
        #     print("  ------------------------------")
        #     print("  train    | {:5d} |  {:8d}   | {:7d}  |".format(num_train_pids, num_train_tracklets, train_num))
        #     print("  query    | {:5d} |  {:8d}   | {:7d}  |".format(num_query_pids, num_query_tracklets, query_num))
        #     print("  gallery  | {:5d} |  {:8d}   | {:7d}  |".format(num_gallery_pids, num_gallery_tracklets, gallery_num))
        #     print("  ------------------------------")
        #     print("  total    | {:5d} |  {:8d}   | {:7d}  |".format(num_total_pids, num_total_tracklets, num_total_imgs))
        #     print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        #     print("  ------------------------------")

        # self.train = train
        self.query = query
        self.gallery = gallery
        # self.valquery = valquery
        # self.valgallery = valgallery

        # self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.tracklets_dir):
            raise RuntimeError("'{}' is not available".format(self.tracklets_dir))

    def _relabel_cam(self):
        self.cam_relabel = []
        id_begin = 0
        for cam_data in self.cam:
            new_data = {}
            for i, otid in enumerate(sorted(cam_data.keys())):
                new_data[i+id_begin] = cam_data[otid]
            self.cam_relabel.append(new_data)
            id_begin = i+id_begin+1

    # def _sample(self):
    #     # time_inter = [20*30, 20*30, 15*30, 10*30, 20*30, 15*30]
    #     cam = self.curr_train_cam
    #     pids_train = []
    #     pids_val = []
    #     for i, cam_data in enumerate(self.cam_relabel):
    #         for pid, tracklets in cam_data.items():
    #             if len(tracklets) < self.min_seq_len_tracklet:
    #                 continue
    #             elif tracklets[0][1] < 46410 and i == cam:
    #                 pids_train.append(pid)
    #             elif tracklets[0][1] >= 46410:
    #                 pids_val.append(pid)
    #         tmp = random.sample(pids_train, self.sample_num)
    #     self.pids_train = tmp
    #     self.pids_val  = pids_val

    def _get_train_tids(self):
        self.train_tids = []
        for camid, cam_data in enumerate(self.cam_relabel):
            tids_tmp = []
            for tid, tracklets in cam_data.items():
                if len(tracklets) < self.min_seq_len_tracklet:
                    continue
                elif tracklets[0][1] < 46410:
                    tids_tmp.append(tid)
            self.train_tids.append(tids_tmp)


    # def update(self):
    #     pids_train = []
    #     self.curr_train_cam = (self.curr_train_cam + 1)%6
    #     cam_data = self.cam_relabel[self.curr_train_cam]
    #     for pid, tracklets in cam_data.items():
    #         if len(tracklets) < self.min_seq_len_train:
    #             continue
    #         elif tracklets[0][1] < 46410:
    #             pids_train.append(pid)
    #     tmp = random.sample(pids_train, 100)
    #     self.pids_train = tmp
    #     train_tracklets = []
    #     num_imgs_per_train_tracklet = []
    #     for pid, tracklets in cam_data.items():
    #         if pid in self.pids_train:
    #             pid_ori = tracklets[0][2]
    #             tdir = osp.join(self.dataset_dir, 'cam%d'%(self.curr_train_cam+1), '%04d'%pid_ori)
    #             img_paths = sorted(glob.glob(osp.join(tdir, '*.jpg')))
    #             img_paths = random.sample(img_paths, self.num_per_person)    #
    #             img_paths = tuple(img_paths)
    #             train_tracklets.append((img_paths, pid, self.curr_train_cam+1))
    #             num_imgs_per_train_tracklet.append(len(img_paths))
    #     self.train = train_tracklets


    def _process_dir_tracklet(self):
        print("=> Automatically generating tracklets")

        self.tracklets = []
        # self.train_num_imgs_per_tracklet = []
        for camid, cam_data in enumerate(self.cam_relabel):
            tracklets_cam = []
            num_imgs_per_tracklet_tmp = []
            for tid, tracklets in cam_data.items():
                if tid in self.train_tids[camid]:
                    tid_ori = tracklets[0][2]
                    tdir = osp.join(self.tracklets_dir, 'cam%d'%(camid+1), '%04d'%tid_ori)
                    img_paths = sorted(glob.glob(osp.join(tdir, '*.jpg')))
                    tmax = int(osp.basename(img_paths[-1])[4:9]) / 30.
                    tmin = int(osp.basename(img_paths[0])[4:9]) / 30.
                    # img_paths = random.sample(img_paths, self.num_per_person)    #
                    # img_paths = tuple(img_paths)
                    tracklets_cam.append((img_paths, tid, camid, tmin, tmax, tid_ori))
                    num_imgs_per_tracklet_tmp.append(len(img_paths))
            self.tracklets.append(tracklets_cam)
            # self.train_num_imgs_per_tracklet.append(num_imgs_per_tracklet_tmp)

    def get_train_tracklets(self):
        train_tracklets = []
        for tracklets_cam in self.tracklets:
            tmp = random.sample(tracklets_cam, self.sample_num)
            new_tracklets_tmp = []
            for tracklet in tmp:
                img_paths, tid, camid, tmin, tmax, tid_ori = tracklet
                img_paths = random.sample(img_paths, self.num_per_person)
                img_paths = tuple(img_paths)
                new_tracklets_tmp.append((img_paths, tid, camid, tmin, tmax, tid_ori))
            # train_tracklets.extend(tmp)
            train_tracklets.extend(new_tracklets_tmp)
        return train_tracklets

    def update_train(self, epoch, feats_cams, camids_cams, tmins_cams, tmaxs_cams, save_dir, k, nosp=False, sigma=0.3, use_flat=False):
        cnt = []
        train_tracklets_s2 = []
        new_pid = 0
        # pair_cams = [(1,2), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5), (5,6)]
        pair_cams = [(1,2), (1,3), (1,4), (1,5), (1,6), (2,3), (2,4), (2,5), (2,6), (3,4), (3,5), (3,6), (4,5), (4,6), (5,6)]
        for gid, pair_cam in enumerate(pair_cams):
            train_tracklets_s2_g = []
            camid_1, camid_2 = pair_cam
            camid_1, camid_2 = camid_1-1, camid_2-1
            feats_1, feats_2 = feats_cams[camid_1], feats_cams[camid_2]
            camids_1, camids_2 = camids_cams[camid_1], camids_cams[camid_2]
            tmins_1, tmins_2 = tmins_cams[camid_1], tmins_cams[camid_2]
            tmaxs_1, tmaxs_2 = tmaxs_cams[camid_1], tmaxs_cams[camid_2]
            m, n = feats_1.size(0), feats_2.size(0)
            distmat = torch.pow(feats_1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(feats_2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, feats_1, feats_2.t())
            distmat = distmat.numpy()
            if nosp:
                p_mat = np.full((m, n), 1.0)
            else:
                delta_t = delta_time(tmins_1, tmaxs_1, tmins_2, tmaxs_2)
                p_mat = compute_sp(delta_t, camids_1, camids_2, sigma, use_flat)
            dist_final = distmat / p_mat
            inds_1 = np.argsort(dist_final,0)
            inds_2 = np.argsort(dist_final,1)
            rank_1 = np.full((m,n), max(m,n))
            rank_2 = np.full((m,n), max(m,n))
            for i in range(m):
                for j in range(n):
                    rank_1[i, inds_2[i,j]] = j
                    rank_2[inds_1[i,j], j] = i
            rank = rank_1 + rank_2
            matched_tracklets_tmp = []
            dist_tmp = []
            for i in range(m):
                for j in range(n):
                    if rank[i,j] == 0:
                        matched_tracklets_tmp.append((i,j))
                        # dist.append((distmat[i,j], dist_final[i,j]))
                        dist_tmp.append((distmat[i,j], dist_final[i,j]))
            kmeans_dist = [x[0] for x in dist_tmp]
            res = kmeans_1d_k(kmeans_dist, k)
            plot_dir = osp.join(save_dir, 'epoch-%d'%(epoch+1))
            if not osp.exists(plot_dir):
                os.makedirs(plot_dir)
            hist_plot(kmeans_dist, osp.join(plot_dir, '%d-%d.png'%(camid_1, camid_2)))

            matched_tracklets = []
            matched_dist = []
            unmatched_tracklets = []
            unmatched_dist = []
            for i, item in enumerate(matched_tracklets_tmp):
                if res[i] == 0:
                    matched_tracklets.append(item)
                    matched_dist.append(dist_tmp[i])
                else:
                    unmatched_tracklets.append(item)
                    unmatched_dist.append(dist_tmp[i])

            cnt.append(len(matched_tracklets))

            if k != 1:
                dst_neg = osp.join(save_dir, 'epoch-%d'%(epoch+1), '%d-%d-neg'%(camid_1, camid_2))
                dist_mean_neg = np.mean([x[0] for x in unmatched_dist])
                if not osp.exists(dst_neg):
                    os.makedirs(dst_neg)
                for pid_tmp, (idx_1, idx_2) in enumerate(unmatched_tracklets):
                    tracklets_1 = self.tracklets[camid_1][idx_1]
                    tracklets_2 = self.tracklets[camid_2][idx_2]
                    img_paths_1, img_paths_2 = tracklets_1[0], tracklets_2[0]
                    dist_o, dist_f = unmatched_dist[pid_tmp]

                    img_len_1, img_len_2 = len(img_paths_1), len(img_paths_2)
                    otid_1, otid_2 = tracklets_1[5], tracklets_2[5]
                    src_1, src_2 = img_paths_1[int(img_len_1/2)], img_paths_2[int(img_len_2/2)]
                    dst_1, dst_2 = src_1.split('/')[-1], src_2.split('/')[-1]
                    dst_1 = '%04d_%.3f_%.3f_%.3f_%05d_%s'%(pid_tmp, dist_o, dist_f, dist_mean_neg, otid_1, dst_1)
                    dst_2 = '%04d_%.3f_%.3f_%.3f_%05d_%s'%(pid_tmp, dist_o, dist_f, dist_mean_neg, otid_2, dst_2)
                    shutil.copy(src_1, osp.join(dst_neg, dst_1))
                    shutil.copy(src_2, osp.join(dst_neg, dst_2))

            dst_pos = osp.join(save_dir, 'epoch-%d'%(epoch+1), '%d-%d-pos'%(camid_1, camid_2))
            dist_mean_pos = np.mean([x[0] for x in matched_dist])
            if not osp.exists(dst_pos):
                os.makedirs(dst_pos)
            for pid_tmp, (idx_1, idx_2) in enumerate(matched_tracklets):
                tracklets_1 = self.tracklets[camid_1][idx_1]
                tracklets_2 = self.tracklets[camid_2][idx_2]
                img_paths_1, img_paths_2 = tracklets_1[0], tracklets_2[0]
                dist_o, dist_f = matched_dist[pid_tmp]

                img_len_1, img_len_2 = len(img_paths_1), len(img_paths_2)
                otid_1, otid_2 = tracklets_1[5], tracklets_2[5]
                src_1, src_2 = img_paths_1[int(img_len_1/2)], img_paths_2[int(img_len_2/2)]
                dst_1, dst_2 = src_1.split('/')[-1], src_2.split('/')[-1]
                dst_1 = '%04d_%.3f_%.3f_%.3f_%05d_%s'%(pid_tmp, dist_o, dist_f, dist_mean_pos, otid_1, dst_1)
                dst_2 = '%04d_%.3f_%.3f_%.3f_%05d_%s'%(pid_tmp, dist_o, dist_f, dist_mean_pos, otid_2, dst_2)
                shutil.copy(src_1, osp.join(dst_pos, dst_1))
                shutil.copy(src_2, osp.join(dst_pos, dst_2))

                img_paths = img_paths_1 + img_paths_2
                train_tracklets_s2_g.append((img_paths, new_pid, gid))
                new_pid += 1
            train_tracklets_s2.append(train_tracklets_s2_g)

        self.train_tracklets_s2 = train_tracklets_s2
        return cnt

    def get_train_tracklet_s2(self):
        train_tracklets = []
        for tracklets_g in self.train_tracklets_s2:
            sample_num = min(self.sample_num_s2, len(tracklets_g))
            # tmp = random.sample(tracklets_g, sample_num)
            tmp = tracklets_g
            new_tracklets_tmp = []
            for tracklet in tmp:
                img_paths, new_pid, gid = tracklet
                img_paths = random.sample(img_paths, self.num_per_person_s2)
                img_paths = tuple(img_paths)
                new_tracklets_tmp.append((img_paths, new_pid, gid))
            # train_tracklets.extend(tmp)
            train_tracklets.extend(new_tracklets_tmp)
        return train_tracklets



    def _process_dir(self, dir_path, json_path, relabel):
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

        print("=> Automatically generating split (might take a while for the first time)")
        pdirs = glob.glob(osp.join(dir_path, '*')) # avoid .DS_Store
        print("Processing {} with {} person identities".format(dir_path, len(pdirs)))

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        tracklets = []
        num_imgs_per_tracklet = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            if relabel: pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, '*'))
            for tdir in tdirs:
                raw_img_paths = sorted(glob.glob(osp.join(tdir, '*.jpg')))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                num_imgs_per_tracklet.append(num_imgs)
                img_paths = raw_img_paths
                # img_paths = []
                # for img_idx in range(num_imgs):
                #     # some tracklet starts from 0002 instead of 0001
                #     img_idx_name = 'F' + str(img_idx+1).zfill(4)
                #     res = glob.glob(osp.join(tdir, '*' + img_idx_name + '*.jpg'))
                #     if len(res) == 0:
                #         print("Warn: index name {} in {} is missing, jump to next".format(img_idx_name, tdir))
                #         continue
                #     img_paths.append(res[0])

                img_name = osp.basename(img_paths[0])
                tmax = int(osp.basename(img_paths[-1])[9:14]) / 30.
                tmin = int(osp.basename(img_paths[0])[9:14]) / 30.
                # naming format: 0001_C1_F00423_3047-1034-0270-0721.jpg
                camid = int(img_name[6]) - 1
                img_paths = tuple(img_paths)

                # tracklets.append((img_paths, pid, camid, tmin, tmax))
                tracklets.append((img_paths, pid, camid))

        num_pids = len(pid_container)
        num_tracklets = len(tracklets)

        print("Saving split to {}".format(json_path))
        split_dict = {
            'tracklets': tracklets,
            'num_tracklets': num_tracklets,
            'num_pids': num_pids,
            'num_imgs_per_tracklet': num_imgs_per_tracklet,
        }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
