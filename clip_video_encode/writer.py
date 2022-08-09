"""save embeddings."""
import os

import fsspec
import numpy as np
import webdataset as wds

from io import BytesIO


class FileWriter:
    """Writes output as files."""
    def __init__(self, output_folder):
        self.output_folder = output_folder

        self.fs, self.output_folder = fsspec.core.url_to_fs(output_folder)

    def write(self, arr, dst_name):
        save_pth = os.path.join(self.output_folder, dst_name)
        with self.fs.open(save_pth, "wb") as f:
            nbp = BytesIO()
            np.save(nbp, arr)
            f.write(nbp.getbuffer())
    def close(self):
        pass


class WebDatasetWriter:
    """Writes output in WebDataset format."""
    def __init__(self, output_folder, shard_id, oom_shard_count, encode_format):
        self.oom_shard_count = oom_shard_count
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=shard_id, oom_shard_count=oom_shard_count
        )
        self.shard_id = shard_id
        fs, output_path = fsspec.core.url_to_fs(output_folder)
        self.tar_fd = fs.open(f"{output_path}/{shard_name}.tar", "wb")
        self.tarwriter = wds.TarWriter(self.tar_fd)
        self.encode_format = encode_format

    def write(self, arr, key):
        """write sample to tars"""
        sample = {"__key__": key, self.encode_format: arr}
        self.tarwriter.write(sample)

    def close(self):
        self.tarwriter.close()
        self.tar_fd.close()
