"""handles input parsing."""
import os
import json
import glob

import pyarrow.parquet as pq
import pyarrow.csv as csv_pq
import pyarrow as pa


class Reader:
    """Parses input into required data.

    Necessary columns (reader will always look for these columns in parquet and csv):
    * videoLoc - location of video either on disc or URL
    * videoID - unique ID of each video, if not provided, ID = index

    Additional special columns:
    * caption - will be saved in separate key.txt file

    anything else - put in key.json metadata file
    """

    def __init__(self, src, meta_columns=None):
        """
        Input:

        src:
            str: path to mp4 file
            str: youtube link
            str: path to txt file with multiple mp4's or youtube links
            list[str]: list with multiple mp4's or youtube links

        meta_columns:
            list[str]: columns of useful metadata to save with videos
        """
        self.columns = ["videoID", "videoLoc"]
        no_dupl_temp = []
        for c in self.columns:
            if c in meta_columns:
                no_dupl_temp.append(c)
                meta_columns.remove(c)

        self.meta_columns = meta_columns if meta_columns is not None else []

        if isinstance(src, str):
            if src.endswith(".txt"):
                df = csv_pq.read_csv(src, read_options=csv_pq.ReadOptions(column_names=["videoLoc"]))
                df = df.add_column(0, "videoID", [list(range(df.num_rows))])  # add ID's
            elif src.endswith(".csv"):
                df = csv_pq.read_csv(src)
            elif src.endswith(".parquet"):
                with open(src, "rb") as f:
                    columns_to_read = self.columns + meta_columns
                    df = pq.read_table(f, columns=columns_to_read)
            else:  # singular video (mp4 or link)
                src = [src]
        if isinstance(src, list):
            df = pa.Table.from_arrays([src], names=["videoLoc"])
            df = df.add_column(0, "videoID", [list(range(df.num_rows))])  # add ID's

        for c in no_dupl_temp:
            self.meta_columns.append(c)
        self.df = df

    def get_data(self):
        vids = self.df["videoLoc"].to_pylist()
        ids = self.df["videoID"]
        meta = dict(  # pylint: disable=consider-using-dict-comprehension
            [(meta, self.df[meta]) for meta in self.meta_columns]
        )
        return vids, ids, meta

def read_shard(tempdir, pass_through_keys=None):
    """
    Extracts shard a tempdir and returns references to files inside

    Input:
        tempdir:
            path to directory containing contents of an opened WebDataset shard with input data
        pass_through_keys:
            extensions we would like to keep from the source shard in the output shard
    """
    if pass_through_keys is None:
        pass_through_keys = []

    vids = sorted(
        [f.split("/")[-1] for f in glob.glob(os.path.join(tempdir, "*.mp4"))]
    )  # TODO: parameterize the video extension

    read_funcs = {
        "json": lambda path: json.load(open(path, "rb")),
        "txt": lambda path: open(path, "r", encoding="UTF-8").read(),
    }

    keys, meta = [x.split(".mp4")[0] for x in vids], []
    for key in keys:
        metadata = {}

        # handles double extensions for weird metadata types f.e. ".optical-flow.npy" vs. ".clip_b.npy" 
        exts = [".".join(f.split(".")[1:]) for f in glob.glob(os.path.join(tempdir, f"{key}.*"))]
        desired_exts = list(set(pass_through_keys).intersection(set(exts)))

        for ext in desired_exts:
            file_path = os.path.join(tempdir, f"{key}.{ext}")
            metadata[ext] = read_funcs[ext](file_path) if ext in read_funcs else open(path, "rb").read()

        meta.append(metadata)

    vids = [os.path.join(tempdir, v) for v in vids]
    return vids, keys, meta
