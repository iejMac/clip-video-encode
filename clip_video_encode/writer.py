"""save embeddings."""
import os
import json

import fsspec
import numpy as np
import webdataset as wds

from io import BytesIO


write_fmt = {
    "mp4": lambda data: data,  # pylint: disable=unnecessary-lambda
    "txt": lambda data: str(data),  # pylint: disable=unnecessary-lambda
    "json": lambda data: json.dumps(data, indent=4),
}


class FileWriter:
    """Writes output as files."""

    def __init__(self, output_folder):
        self.output_folder = output_folder

        self.fs, self.output_folder = fsspec.core.url_to_fs(output_folder)

    def write(self, arr, key, metadata=None):
        """write sample to file."""
        key, metadata = str(key), {} if metadata is None else metadata

        save_pth = os.path.join(self.output_folder, key + ".npy")
        with self.fs.open(save_pth, "wb") as f:
            nbp = BytesIO()
            np.save(nbp, arr)
            f.write(nbp.getbuffer())

        for ext in metadata:
            md_filename = os.path.join(self.output_folder, f"{key}.{ext}")
            write_data = write_fmt[ext](metadata[ext]) if ext in write_fmt else metadata[ext]
            with self.fs.open(md_filename, "w") as f:
                f.write(write_data)

    def close(self):
        pass


class WebDatasetWriter:
    """Writes output in WebDataset format."""

    def __init__(self, output_folder, oom_shard_count, encode_format, maxcount=10000, shard_id=0):
        self.output_folder = output_folder
        self.oom_shard_count = oom_shard_count
        self.encode_format = encode_format
        self.maxcount = maxcount
        self.shard_id = shard_id
        self.shard_suffix = "clip_embeddings"  # TODO: maybe there should be param for this?

        self.count = 0

        self.tarwriter = None
        self.tar_fd = None

        self.create_shard()

    def create_shard(self, shard_id=None):
        """create new shard in sequential order."""
        self.close()
        if shard_id is not None:
            self.shard_id = shard_id
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=self.shard_id, oom_shard_count=self.oom_shard_count
        )
        shard_name += "_" + self.shard_suffix

        fs, output_path = fsspec.core.url_to_fs(self.output_folder)
        self.tar_fd = fs.open(f"{output_path}/{shard_name}.tar", "wb")
        self.tarwriter = wds.TarWriter(self.tar_fd)

    def write(self, arr, key, metadata=None):
        """write sample to current shard."""
        key, metadata = str(key), {} if metadata is None else metadata
        if self.count >= self.maxcount:
            self.shard_id += 1
            self.count = 0
            self.create_shard()

        sample = {"__key__": key}
        if arr is not None:
            sample[self.encode_format] = arr

        for ext in metadata:
            sample[ext] = write_fmt[ext](metadata[ext]) if ext in write_fmt else metadata[ext]

        self.tarwriter.write(sample)
        self.count += 1

    def close(self):
        if self.tarwriter is not None:
            self.tarwriter.close()
            self.tar_fd.close()
