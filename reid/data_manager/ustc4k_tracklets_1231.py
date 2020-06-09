from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import copy
import re
import pickle
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave

from reid.utils.iotools import mkdir_if_missing, write_json, read_json

def unpkl(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

class USTC4k_tracklets_1231(object):
    """
    USTC4kVidReID

    Dataset statistics:
    # identities: 
    # tracklets:
    """
    cam = []
    for i in range(6):
        data = unpkl('cam%d_filtered.pkl'%(i+1))
        cam.append(data)

    def __init__(self, root, min_seq_len=0, verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, 'ustc4k-tracklets-1231')
        self.test_dir = osp.join(root, 'ustc4k-181213')

        self.split_train_json_path = osp.join(self.dataset_dir, 'split_train.json')
        self.split_valquery_json_path = osp.join(self.dataset_dir, 'split_valquery.json')
        self.split_valgallery_json_path = osp.join(self.dataset_dir, 'split_valgallery.json')
        # self.split_train_json_path = osp.join(self.test_dir, 'split_train.json')
        self.split_query_json_path = osp.join(self.dataset_dir, 'split_query.json')
        self.split_gallery_json_path = osp.join(self.dataset_dir, 'split_gallery.json')

        self.min_seq_len = min_seq_len
        self.min_seq_len_train = 0
        self._check_before_run()
        self._relabel_cam()
        self._sample()
        self._process_dir_train(self.dataset_dir, True)

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
            self.train_split_dict['tracklets'], self.train_split_dict['num_tracklets'], self.train_split_dict['num_pids'], self.train_split_dict['num_imgs_per_tracklet']
        valquery, num_valquery_tracklets, num_valquery_pids, num_imgs_valquery = \
            self.valquery_split_dict['tracklets'], self.valquery_split_dict['num_tracklets'], self.valquery_split_dict['num_pids'], self.valquery_split_dict['num_imgs_per_tracklet']
        valgallery, num_valgallery_tracklets, num_valgallery_pids, num_imgs_valgallery = \
            self.valgallery_split_dict['tracklets'], self.valgallery_split_dict['num_tracklets'], self.valgallery_split_dict['num_pids'], self.valgallery_split_dict['num_imgs_per_tracklet']
        # self.train_dir = osp.join(self.test_dir, 'train')
        self.query_dir = osp.join(self.test_dir, 'query')
        self.gallery_dir = osp.join(self.test_dir, 'gallery')
        # train, num_train_tracklets, num_train_pids, num_imgs_train = \
        #     self._process_dir(self.train_dir, self.split_train_json_path, relabel=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
            self._process_dir(self.query_dir, self.split_query_json_path, relabel=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
            self._process_dir(self.gallery_dir, self.split_gallery_json_path, relabel=False)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        train_num = np.sum(num_imgs_train)
        query_num = np.sum(num_imgs_query)
        gallery_num = np.sum(num_imgs_gallery)
        num_total_imgs = train_num + query_num + gallery_num
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print(num_query_pids, num_query_tracklets, query_num)

        if verbose:
            print("=> USTC4k_181213 loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # tracklets |  # imgs  |")
            print("  ------------------------------")
            print("  train    | {:5d} |  {:8d}   | {:7d}  |".format(num_train_pids, num_train_tracklets, train_num))
            print("  query    | {:5d} |  {:8d}   | {:7d}  |".format(num_query_pids, num_query_tracklets, query_num))
            print("  gallery  | {:5d} |  {:8d}   | {:7d}  |".format(num_gallery_pids, num_gallery_tracklets, gallery_num))
            print("  ------------------------------")
            print("  total    | {:5d} |  {:8d}   | {:7d}  |".format(num_total_pids, num_total_tracklets, num_total_imgs))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery
        self.valquery = valquery
        self.valgallery = valgallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids


        # train, num_train_tracklets, num_train_pids, num_imgs_train, \
        # query, num_query_tracklets, num_query_pids, num_imgs_query, \
        # gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
        #     self._process_dir(self.dataset_dir, self.split_train_json_path, self.split_query_json_path, self.split_gallery_json_path, relabel=True)



    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _relabel_cam(self):
        self.cam_relabel = []
        id_begin = 0
        for i, cam_data in enumerate(self.cam):
            new_data = {}
            for i, pid in enumerate(sorted(cam_data.keys())):
                new_data[i+id_begin] = cam_data[pid]
            self.cam_relabel.append(new_data)
            id_begin = i+id_begin+1

    def _sample(self):
        # time_inter = [20*30, 20*30, 15*30, 10*30, 20*30, 15*30]
        pids_train = []
        pids_val = []
        for i, cam_data in enumerate(self.cam_relabel):
            for pid, tracklets in cam_data.items():
                if len(tracklets) < self.min_seq_len_train:
                    continue
                elif tracklets[0][1] < 46410 and i == 4:
                    pids_train.append(pid)
                elif tracklets[0][1] >= 46410:
                    pids_val.append(pid)
        self.pids_train = pids_train
        self.pids_val  = pids_val

    def _process_dir_train(self, dir_path, relabel):
        # if osp.exists(self.split_train_json_path) and osp.exists(self.split_valquery_json_path) and osp.exists(self.split_valgallery_json_path):
        #     print("=> {} generated before, awesome!".format(self.split_train_json_path))
        #     self.train_split_dict = read_json(self.split_train_json_path)
        #     self.valquery_split_dict = read_json(self.split_valquery_json_path)
        #     self.valgallery_split_dict = read_json(self.split_valgallery_json_path)
        #     return

        print("=> Automatically generating split train (might take a while for the first time)")

        train_tracklets = []
        valquery_tracklets = []
        valgallery_tracklets = []
        num_imgs_per_train_tracklet = []
        num_imgs_per_valquery_tracklet = []
        num_imgs_per_valgallery_tracklet = []
        for i, cam_data in enumerate(self.cam_relabel):
            for pid, tracklets in cam_data.items():
                if pid in self.pids_train:
                    pid_ori = tracklets[0][2]
                    tdir = osp.join(dir_path, 'cam%d'%(i+1), '%04d'%pid_ori)
                    img_paths = sorted(glob.glob(osp.join(tdir, '*.jpg')))
                    img_paths = tuple(img_paths)
                    train_tracklets.append((img_paths, pid, i+1))
                    num_imgs_per_train_tracklet.append(len(img_paths))
                elif pid in self.pids_val:
                    pid_ori = tracklets[0][2]
                    tdir = osp.join(dir_path, 'cam%d'%(i+1), '%04d'%pid_ori)
                    raw_img_paths = sorted(glob.glob(osp.join(tdir, '*.jpg')))
                    img_paths_query = raw_img_paths[:len(raw_img_paths)//2]
                    img_paths_gallery = raw_img_paths[len(raw_img_paths)//2:]
                    img_paths_query = tuple(img_paths_query)
                    img_paths_gallery = tuple(img_paths_gallery)
                    valquery_tracklets.append((img_paths_query, pid, 0))
                    valgallery_tracklets.append((img_paths_gallery, pid, i+1))
                    num_imgs_per_valquery_tracklet.append(len(img_paths_query))
                    num_imgs_per_valgallery_tracklet.append(len(img_paths_gallery))

        print("saving split to %s, %s, %s"%(self.split_train_json_path, self.split_valquery_json_path, self.split_valgallery_json_path))
        self.train_split_dict = {
            'tracklets': train_tracklets,
            'num_tracklets': len(train_tracklets),
            'num_pids': len(train_tracklets),
            'num_imgs_per_tracklet': num_imgs_per_train_tracklet,
        }
        self.valquery_split_dict = {
            'tracklets': valquery_tracklets,
            'num_tracklets': len(valquery_tracklets),
            'num_pids': len(valquery_tracklets),
            'num_imgs_per_tracklet': num_imgs_per_valquery_tracklet,
        }
        self.valgallery_split_dict = {
            'tracklets': valgallery_tracklets,
            'num_tracklets': len(valgallery_tracklets),
            'num_pids': len(valgallery_tracklets),
            'num_imgs_per_tracklet': num_imgs_per_valgallery_tracklet,
        }
        write_json(self.train_split_dict, self.split_train_json_path)
        write_json(self.valquery_split_dict, self.split_valquery_json_path)
        write_json(self.valgallery_split_dict, self.split_valgallery_json_path)

    def _process_dir(self, dir_path, json_path, relabel):
        # if osp.exists(json_path):
        #     print("=> {} generated before, awesome!".format(json_path))
        #     split = read_json(json_path)
        #     return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

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
                # naming format: 0001_C1_F00423_3047-1034-0270-0721.jpg
                camid = int(img_name[6]) - 1
                img_paths = tuple(img_paths)
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


        


