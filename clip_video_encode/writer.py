"""save embeddings."""
import os

import fsspec
import numpy as np


class FileWriter:
    """Writes output as files."""
    def __init__(self, dest_dir):
        self.dest_dir = dest_dir

        self.fs, self.dest_dir = fsspec.core.url_to_fs(dest_dir)

    def write(self, arr, dst_name):
        save_pth = os.path.join(self.dest_dir, dst_name)
        with self.fs.open(save_pth, "wb") as f:
            nbp = BytesIO()
            np.save(nbp, arr)
            f.write(nbp.getbuffer())


class WebDatasetWriter:
    """Writes output in WebDataset format."""
    def __init__(self):
        pass

