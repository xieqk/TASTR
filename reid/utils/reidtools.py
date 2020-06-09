from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import os.path as osp
import shutil

from .iotools import mkdir_if_missing


def visualize_ranked_results(distmat, dataset, save_dir='log/ranked_results', topk=20):
    """
    Visualize ranked results

    Support both imgreid and vidreid

    Args:
    - distmat: distance matrix of shape (num_query, num_gallery).
    - dataset: has dataset.query and dataset.gallery, both are lists of (img_path, pid, camid);
               for imgreid, img_path is a string, while for vidreid, img_path is a tuple containing
               a sequence of strings.
    - save_dir: directory to save output images.
    - topk: int, denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape

    print("Visualizing top-{} ranks".format(topk))
    print("# query: {}\n# gallery {}".format(num_q, num_g))
    print("Saving images to '{}'".format(save_dir))
    
    assert num_q == len(dataset.query)
    assert num_g == len(dataset.gallery)
    
    indices = np.argsort(distmat, axis=1)
    mkdir_if_missing(save_dir)

    def _cp_img_to(src, dst, rank, prefix):
        """
        - src: image path or tuple (for vidreid)
        - dst: target directory
        - rank: int, denoting ranked position, starting from 1
        - prefix: string
        """
        src_base = osp.basename(src)
        this_pid, this_camid, this_fid = src_base.split('_')[0], src_base.split('_')[1], src_base.split('_')[2]
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + 'top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + 'top' + str(rank).zfill(2) + '_%s-%s-%s.jpg'%(this_pid, this_camid, this_fid))
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = dataset.query[q_idx]

        # my add
        qimg_path = qimg_path[int(len(qimg_path)/2)]
        this_pid, this_tid = qimg_path.split('/')[-3], qimg_path.split('/')[-2]
        qdir = osp.join(save_dir, '%s-%s-C%d'%(this_pid, this_tid, qcamid))

        # qdir = osp.join(save_dir, osp.basename(qimg_path))
        mkdir_if_missing(qdir)
        _cp_img_to(qimg_path, qdir, rank=0, prefix='')

        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            gimg_path, gpid, gcamid = dataset.gallery[g_idx]

            # my add
            gimg_path = gimg_path[int(len(gimg_path)/2)]
            invalid = False

            # invalid = (qpid == gpid) & (qcamid == gcamid)
            if not invalid:
                _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='')
                rank_idx += 1
                if rank_idx > topk:
                    break

    print("Done")
