"""encode numpy video frame arrays with CLIP from directory as they come in from other processes."""
import os
import time

import numpy as np

from .utils import block2dl

N_DATASET_WORKERS = 6
BATCH_SIZE = 256


class LiveNumpyEncoder:
    """class that watches directory for set of numpy arrays of videos to encode using CLIP."""

    def __init__(self, data_dir, dest_dir, vids, mapper, preprocess):
        """

        Input:
            data_dir: directory to watch for np files
            dest_dir:  where to save embeddings to
            vids: list of numpy array names to watch for (completes when all fnmaes have been seen).
                  JUST "NAME.npy", NOT FULL PATH
            mapper: model used to map frames to embeddings
            preprocess: function to preprocess the frames with
        """
        assert data_dir != dest_dir  # input and output will have same name
        self.data_dir = data_dir
        self.dest_dir = dest_dir
        self.vids = vids

        self.fm = mapper
        self.preprocess = preprocess

    def start(self):
        """starts live reading."""
        while len(self.vids) > 0:  # haven't seen all videos.
            # TODO: decide if we need some checks here for incorrectly placed files
            available_vids = os.listdir(self.data_dir)  # for now assuming all vids in self.data_dir are correct
            if len(available_vids) == 0:
                print("Waiting for arrays...")
                time.sleep(5)  # wait for arrays to come in
                continue

            np_arrays = []
            name_inds = []
            cur_len = 0
            for vid in available_vids:
                assert vid.endswith(".npy")
                vid_frames = np.load(os.path.join(self.data_dir, vid))
                name_inds.append((vid, cur_len, cur_len + len(vid_frames)))
                cur_len += len(vid_frames)
                np_arrays.append(vid_frames)

                self.vids.remove(vid)

            frame_chunk = np.concatenate(np_arrays)
            dl = block2dl(frame_chunk, self.preprocess, BATCH_SIZE, N_DATASET_WORKERS)

            embeddings = []
            for batch in dl:
                emb = self.fm(batch.to(self.fm.device))
                embeddings.append(emb)

            all_embs = np.concatenate(embeddings)
            for name, i0, it in name_inds:
                vid_embs = all_embs[i0:it]
                save_pth = os.path.join(self.dest_dir, name)
                np.save(save_pth, vid_embs)
