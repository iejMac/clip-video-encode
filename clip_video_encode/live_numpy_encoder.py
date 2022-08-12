"""encode numpy video frame arrays with CLIP from directory as they come in from other processes."""
import os
import time

import numpy as np

from .utils import block2dl
from .writer import FileWriter

N_DATASET_WORKERS = 6
BATCH_SIZE = 256


class LiveNumpyEncoder:
    """class that watches directory for set of numpy arrays of videos to encode using CLIP."""

    def __init__(self, data_dir, dest_dir, n_vids, mapper, preprocess, frame_mem=4, remove_on_read=False):
        """

        Input:
            data_dir: directory to watch for np files
            dest_dir:  where to save embeddings to
            n_vids: number of numpy array names to watch for. Completes after n_vids have been encoded
            mapper: model used to map frames to embeddings
            preprocess: function to preprocess the frames with
            frame_mem: amount of memory in GB for shared frame array
            remove_on_read: remove arrays when done reading them
        """
        assert data_dir != dest_dir  # input and output will have same name
        self.data_dir = data_dir
        self.writer = FileWriter(dest_dir)
        self.n_vids = n_vids
        self.frame_mem = frame_mem

        self.fm = mapper
        self.preprocess = preprocess

        self.remove_on_read = remove_on_read

    def start(self):
        """starts live reading."""

        mem_size_b = int(self.frame_mem * 1024**3)
        mem_frames = mem_size_b // (224**2 * 3)
        frame_array = np.zeros((mem_frames, 224, 224, 3), dtype=np.uint8)
        embedding_array = np.zeros((mem_frames, 512))

        while self.n_vids > 0:  # haven't seen all videos.
            # TODO: decide if we need some checks here for incorrectly placed files
            available_vids = os.listdir(self.data_dir)  # for now assuming all vids in self.data_dir are correct
            if len(available_vids) == 0:
                print("Waiting for arrays...")
                time.sleep(5)  # wait for arrays to come in
                continue

            print(f"Found {len(available_vids)} arrays.")

            name_inds = []

            t0 = time.perf_counter()

            cur_len = 0
            for vid in available_vids:
                assert vid.endswith(".npy")
                vid_path = os.path.join(self.data_dir, vid)
                vid_frames = np.load(vid_path)
                frame_array[cur_len : cur_len + vid_frames.shape[0]] = vid_frames
                name_inds.append((vid, cur_len, cur_len + vid_frames.shape[0]))
                cur_len += vid_frames.shape[0]

                self.n_vids -= 1
                if self.remove_on_read:
                    os.remove(vid_path)

            t_load = time.perf_counter() - t0
            print(f"Load time: {t_load}")

            t0 = time.perf_counter()

            frame_chunk = frame_array[:cur_len]
            dl = block2dl(frame_chunk, self.preprocess, BATCH_SIZE, N_DATASET_WORKERS)

            cur_len = 0
            for batch in dl:
                emb = self.fm(batch.to(self.fm.device))
                embedding_array[cur_len : cur_len + emb.shape[0]] = emb
                cur_len += emb.shape[0]

            t_enc = time.perf_counter() - t0
            print(f"Encode time: {t_enc}")

            all_embs = embedding_array[:cur_len]

            for name, i0, it in name_inds:
                self.writer.write(all_embs[i0:it], name)
