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
from scipy.misc import imsave

from reid.utils.iotools import mkdir_if_missing, write_json, read_json

def unpkl(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

class USTC4k_tracklets_s2(object):
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
            data = unpkl('cam%d_filtered.pkl'%(i+1))
            print(i, ':', len(data.keys()))
            self.cam.append(data)
        self.min_seq_len = min_seq_len
        self.curr_train_cam = 0
        self.sample_num = 100
        self.num_per_person = 60
        self.min_seq_len_tracklet = 60
        assert self.num_per_person <= self.min_seq_len_tracklet
        self.tracklets_dir = osp.join(root, 'ustc4k-tracklets-1231')
        self.dataset_dir = osp.join(root, 'ustc4k-181213')

        self.split_train_json_path = osp.join(self.tracklets_dir, '0111_split_train.json')
        self.split_valquery_json_path = osp.join(self.tracklets_dir, '0111_split_valquery.json')
        self.split_valgallery_json_path = osp.join(self.tracklets_dir, '0111_split_valgallery.json')
        # self.split_train_json_path = osp.join(self.test_dir, 'split_train.json')
        self.split_query_json_path = osp.join(self.dataset_dir, '0111_split_query.json')
        self.split_gallery_json_path = osp.join(self.dataset_dir, '0111_split_gallery.json')
        
        self._check_before_run()
        self._relabel_cam()
        self._get_train_tids()
        # self._sample()
        self._process_dir_tracklet()
        # self._process_dir_dataset()

        # train, num_train_tracklets, num_train_pids, num_imgs_train = \
        #     self.train_split_dict['tracklets'], self.train_split_dict['num_tracklets'], self.train_split_dict['num_pids'], self.train_split_dict['num_imgs_per_tracklet']
        # valquery, num_valquery_tracklets, num_valquery_pids, num_imgs_valquery = \
        #     self.valquery_split_dict['tracklets'], self.valquery_split_dict['num_tracklets'], self.valquery_split_dict['num_pids'], self.valquery_split_dict['num_imgs_per_tracklet']
        # valgallery, num_valgallery_tracklets, num_valgallery_pids, num_imgs_valgallery = \
        #     self.valgallery_split_dict['tracklets'], self.valgallery_split_dict['num_tracklets'], self.valgallery_split_dict['num_pids'], self.valgallery_split_dict['num_imgs_per_tracklet']
        # # self.train_dir = osp.join(self.test_dir, 'train')
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
        for i, cam_data in enumerate(self.cam):
            new_data = {}
            for i, tid in enumerate(sorted(cam_data.keys())):
                new_data[i+id_begin] = cam_data[tid]
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


    def update(self):
        pids_train = []
        self.curr_train_cam = (self.curr_train_cam + 1)%6
        cam_data = self.cam_relabel[self.curr_train_cam]
        for pid, tracklets in cam_data.items():
            if len(tracklets) < self.min_seq_len_train:
                continue
            elif tracklets[0][1] < 46410:
                pids_train.append(pid)
        tmp = random.sample(pids_train, 100)
        self.pids_train = tmp
        train_tracklets = []
        num_imgs_per_train_tracklet = []
        for pid, tracklets in cam_data.items():
            if pid in self.pids_train:
                pid_ori = tracklets[0][2]
                tdir = osp.join(self.dataset_dir, 'cam%d'%(self.curr_train_cam+1), '%04d'%pid_ori)
                img_paths = sorted(glob.glob(osp.join(tdir, '*.jpg')))
                img_paths = random.sample(img_paths, self.num_per_person)    #
                img_paths = tuple(img_paths)
                train_tracklets.append((img_paths, pid, self.curr_train_cam+1))
                num_imgs_per_train_tracklet.append(len(img_paths))
        self.train = train_tracklets


    def _process_dir_tracklet(self):
        print("=> Automatically generating tracklets")

        self.train_tracklets = []
        self.train_num_imgs_per_tracklet = []
        for camid, cam_data in enumerate(self.cam_relabel):
            tracklets_tmp = []
            num_imgs_per_tracklet_tmp = []
            for tid, tracklets in cam_data.items():
                if tid in self.train_tids[camid]:
                    tid_ori = tracklets[0][2]
                    tdir = osp.join(self.tracklets_dir, 'cam%d'%(camid+1), '%04d'%tid_ori)
                    img_paths = sorted(glob.glob(osp.join(tdir, '*.jpg')))
                    tmax = int(osp.basename(img_paths[-1])[4:9]) / 30
                    tmin = int(osp.basename(img_paths[0])[4:9]) / 30
                    # img_paths = random.sample(img_paths, self.num_per_person)    #
                    img_paths = tuple(img_paths)
                    tracklets_tmp.append((img_paths, tid, camid, tmin, tmax, tid_ori))
                    num_imgs_per_tracklet_tmp.append(len(img_paths))
            self.train_tracklets.append(tracklets_tmp)
            self.train_num_imgs_per_tracklet.append(num_imgs_per_tracklet_tmp)


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
                tmax = int(osp.basename(img_paths[-1])[9:14]) / 30
                tmin = int(osp.basename(img_paths[0])[9:14]) / 30
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


        


