"""
processes s3://s-datasets/kinetics-700/kinetics700_embeddings into processed dataset format.

run from kinetics700_embeddings directory (so train/val/test at the same level)
"""

import os
import glob
import json
import shutil

from tqdm import tqdm

SPLITS = ["train", "val", "test"]
PROCESSED_DIR = "processed"

sample_id = 0
for split in SPLITS:
    print(f"Processing split - {split}")
    npys = glob.glob(os.path.join(split, "**/*.npy"))

    os.makedirs(os.path.join(PROCESSED_DIR, split), exist_ok=True)
    for npy in tqdm(npys):
        _, cap, meta_string = npy.split("/")

        videoID, start_t, end_t = meta_string[:11], meta_string[12:18], meta_string[19:-4]
        meta = {
            "videoID": videoID,
            "start_time": start_t,
            "end_time": end_t,
        }

        fname_prefix = os.path.join(PROCESSED_DIR, split, f"vid_{sample_id:09}")

        with open(fname_prefix + ".txt", "w", encoding="utf-8") as f:
            f.write(cap)  # for Kinetics700 caption is label
        with open(fname_prefix + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        shutil.copyfile(npy, fname_prefix + ".npy")

        sample_id += 1  # UNIQUE ID FOR EACH VIDEO
