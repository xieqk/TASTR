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

import torch

from reid.utils.iotools import mkdir_if_missing, write_json, read_json
from reid.utils.sp_utils import *


class DukeMTMC_reID(object):
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
        self.dist_mat = np.array([
            [0, 25, 47, 90, 52, 83, 78, 25],
            [25, 0, 6, 44, 10, 40, 35, 26],
            [47, 6, 0, 20, 0, 36, 33, 50],
            [90, 44, 20, 0, 32, 65, 63, 88],
            [52, 10, 0, 32, 0, 14, 9, 40],
            [83, 40, 36, 65, 14, 0, 8, 50],
            [78, 35, 33, 63, 9, 8, 0, 25],
            [25, 26, 50, 88, 40, 50, 25, 0]
        ])
        self.time_mat = self.dist_mat / 1.25
        self.time_bias = [5542, 3606, 27243, 31181, 0, 22401, 18967, 46765]
        # self.cam = []
        # for i in range(6):
        #     data = unpkl('cam%d_filtered.pkl'%(i+1))
        #     print(i, ':', len(data.keys()))
        #     self.cam.append(data)
        # self.min_seq_len = min_seq_len
        # self.curr_train_cam = 0
        # self.sample_num = 100
        # self.sample_num_s2 = 50
        # self.num_per_person = 60
        # self.num_per_person_s2 = 60
        # self.min_seq_len_tracklet = 60
        # assert self.num_per_person <= self.min_seq_len_tracklet
        self.dataset_dir = osp.join(root, 'DukeMTMC-reID')
        self.tracklets_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        # self.split_query_json_path = osp.join(self.dataset_dir, 'split_query.json')
        # self.split_gallery_json_path = osp.join(self.dataset_dir, 'split_gallery.json')

        self._check_before_run()
        # self._sample()
        self._process_dir_tracklet()
        # self._process_dir_dataset()

        # # train, num_train_tracklets, num_train_pids, num_imgs_train = \
        # #     self._process_dir(self.train_dir, self.split_train_json_path, relabel=True)
        query, num_query_pids, num_imgs_query = \
            self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_imgs_gallery = \
            self._process_dir(self.gallery_dir, relabel=False)

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

    def _process_dir_tracklet(self):
        print("=> Automatically generating tracklets")

        self.tracklets = [[], [], [], [], [], [], [], []]
        img_list = sorted(glob.glob(osp.join(self.tracklets_dir, '*.jpg')))

        tracklet_img_paths = []
        new_tid = 0
        ts = []

        img_basename = osp.basename(img_list[0]).split('.')[0]
        pid_str, camid_str, frameid_str = img_basename.split('_')
        _pid, _camid, _frameid = int(pid_str), int(camid_str[1])-1, int(frameid_str[1:])
        for img_path in img_list:
            img_basename = osp.basename(img_path).split('.')[0]
            pid_str, camid_str, frameid_str = img_basename.split('_')
            pid, camid, frameid = int(pid_str), int(camid_str[1])-1, int(frameid_str[1:])
            cur_t = (frameid + self.time_bias[camid]) / 60.0
            if pid != _pid or camid != _camid:
                self.tracklets[_camid].append((tracklet_img_paths, new_tid, _camid,
                min(ts), max(ts), pid_str))
                tracklet_img_paths = []
                new_tid += 1
                ts = []
                _pid, _camid, _frameid = pid, camid, frameid

            ts.append(cur_t)
            tracklet_img_paths.append(img_path)

    def get_train_tracklets(self):
        train_tracklets = []
        for tracklets_cam in self.tracklets:
            # tmp = random.sample(tracklets_cam, self.sample_num)
            tmp = tracklets_cam
            new_tracklets_tmp = []
            for tracklet in tmp:
                img_paths, tid, camid, tmin, tmax, tid_ori = tracklet
                # img_paths = random.sample(img_paths, self.num_per_person)
                img_paths = tuple(img_paths)
                new_tracklets_tmp.append((img_paths, tid, camid, tmin, tmax, tid_ori))
            # train_tracklets.extend(tmp)
            train_tracklets.extend(new_tracklets_tmp)
        return train_tracklets

    def update_train(self, epoch, feats_cams, camids_cams, tmins_cams, tmaxs_cams, save_dir, k, nosp=False, sigma=0.7, use_flat=False):
        cnt = []
        cnt_true = []
        train_tracklets_s2 = []
        new_pid = 0
        # pair_cams = [(1,2), (1,8), (2,3), (2,5), (3,4), (3,5), (4,5), (5,6), (5,7), (6,7), (7,8)]
        pair_cams = [(1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (3,4), (3,5), (3,6), (3,7), (3,8), (4,5), (4,6), (4,7), (4,8), (5,6), (5,7), (5,8), (6,7), (6,8), (7,8)]
        print(pair_cams)
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
                print('no-sp')
                p_mat = np.full((m, n), 1.0)
            else:
                delta_t = delta_time(tmins_1, tmaxs_1, tmins_2, tmaxs_2)
                # p_mat = compute_sp(delta_t, camids_1, camids_2, sigma, use_flat)
                p_mat = compute_sp(delta_t, camids_1, camids_2, sigma, use_flat, time_mat=self.time_mat)
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
            if not nosp:
                dt_tmp = []
            for i in range(m):
                for j in range(n):
                    if rank[i,j] == 0:
                        matched_tracklets_tmp.append((i,j))
                        # dist.append((distmat[i,j], dist_final[i,j]))
                        dist_tmp.append((distmat[i,j], dist_final[i,j]))
                        if not nosp:
                            dt_tmp.append(delta_t[i,j])
            kmeans_dist = [x[0] for x in dist_tmp]
            res = kmeans_1d_k(kmeans_dist, k)
            # res = np.zeros(res.shape) ####################################
            plot_dir = osp.join(save_dir, 'epoch-%d'%(epoch+1))
            if not osp.exists(plot_dir):
                os.makedirs(plot_dir)
            # hist_plot(kmeans_dist, osp.join(plot_dir, '%d-%d.png'%(camid_1, camid_2)))

            matched_tracklets = []
            matched_dist = []
            unmatched_tracklets = []
            unmatched_dist = []
            matched_dt = []
            for i, item in enumerate(matched_tracklets_tmp):
                if res[i] == 0:
                    matched_tracklets.append(item)
                    matched_dist.append(dist_tmp[i])
                    if not nosp:
                        matched_dt.append(dt_tmp[i])
                else:
                    unmatched_tracklets.append(item)
                    unmatched_dist.append(dist_tmp[i])

            dt_mean = np.array(matched_dt).mean()
            # self.time_mat[camid_1, camid_2] = 0.8*self.time_mat[camid_1, camid_2] + 0.2*dt_mean
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
                    dst_1 = '%04d_%.3f_%.3f_%.3f_%s_%s'%(pid_tmp, dist_o, dist_f, dist_mean_neg, otid_1, dst_1)
                    dst_2 = '%04d_%.3f_%.3f_%.3f_%s_%s'%(pid_tmp, dist_o, dist_f, dist_mean_neg, otid_2, dst_2)
                    # shutil.copy(src_1, osp.join(dst_neg, dst_1))
                    # shutil.copy(src_2, osp.join(dst_neg, dst_2))

            dst_pos = osp.join(save_dir, 'epoch-%d'%(epoch+1), '%d-%d-pos'%(camid_1, camid_2))
            dist_mean_pos = np.mean([x[0] for x in matched_dist])
            if not osp.exists(dst_pos):
                os.makedirs(dst_pos)
            cnt_true_tmp = 0
            for pid_tmp, (idx_1, idx_2) in enumerate(matched_tracklets):
                tracklets_1 = self.tracklets[camid_1][idx_1]
                tracklets_2 = self.tracklets[camid_2][idx_2]
                img_paths_1, img_paths_2 = tracklets_1[0], tracklets_2[0]
                dist_o, dist_f = matched_dist[pid_tmp]

                img_len_1, img_len_2 = len(img_paths_1), len(img_paths_2)
                otid_1, otid_2 = tracklets_1[5], tracklets_2[5]
                if int(otid_1) == int(otid_2):
                    cnt_true_tmp += 1
                src_1, src_2 = img_paths_1[int(img_len_1/2)], img_paths_2[int(img_len_2/2)]
                dst_1, dst_2 = src_1.split('/')[-1], src_2.split('/')[-1]
                dst_1 = '%04d_%.3f_%.3f_%.3f_%s_%s'%(pid_tmp, dist_o, dist_f, dist_mean_pos, otid_1, dst_1)
                dst_2 = '%04d_%.3f_%.3f_%.3f_%s_%s'%(pid_tmp, dist_o, dist_f, dist_mean_pos, otid_2, dst_2)
                # shutil.copy(src_1, osp.join(dst_pos, dst_1))
                # shutil.copy(src_2, osp.join(dst_pos, dst_2))

                img_paths = img_paths_1 + img_paths_2
                train_tracklets_s2_g.append((img_paths, new_pid, gid))
                new_pid += 1
            train_tracklets_s2.append(train_tracklets_s2_g)
            cnt_true.append(cnt_true_tmp)
        self.train_tracklets_s2 = train_tracklets_s2
        return cnt, cnt_true

    def get_train_tracklet_s2(self):
        train_tracklets = []
        for tracklets_g in self.train_tracklets_s2:
            # sample_num = min(self.sample_num_s2, len(tracklets_g))
            # tmp = random.sample(tracklets_g, sample_num)
            tmp = tracklets_g
            new_tracklets_tmp = []
            for tracklet in tmp:
                img_paths, new_pid, gid = tracklet
                # img_paths = random.sample(img_paths, self.num_per_person_s2)
                img_paths = tuple(img_paths)
                new_tracklets_tmp.append((img_paths, new_pid, gid))
            # train_tracklets.extend(tmp)
            train_tracklets.extend(new_tracklets_tmp)
        return train_tracklets



    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs
