"""save embeddings block using vid_inds dict"""
import os

import numpy as np


def write_embeddings(vid_inds, embeddings, dst=""):
    for dst_name, inds in vid_inds.items():
        i0, it = inds
        vid_embeddings = embeddings[i0:it]
        save_pth = os.path.join(dst, dst_name)
        np.save(save_pth, vid_embeddings)
