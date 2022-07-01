"""creates EmbeddingWebDataset from Processed Dataset format"""
import os
import os.path
import random
import argparse
import json

from csv import writer
from pathlib import Path

import numpy as np
import webdataset as wds

parser = argparse.ArgumentParser("""Generate Embedding WebDataset from Processed Dataset.""")
parser.add_argument("--maxsize", type=float, default=1e9)
parser.add_argument("--maxcount", type=float, default=10000)
parser.add_argument(
    "--compression",
    dest="compression",
    action="store_true",
    help="Creates compressed .tar.gz files instead of uncompressed .tar files.",
)
parser.add_argument("--json", dest="json", action="store_true", help="Reads json files and add them to the .tar files.")
parser.add_argument("--shards", default="./shards", help="directory where shards are written")
parser.add_argument(
    "--data",
    default="./data",
    help="directory path containing Processed Dataset",
)
args = parser.parse_args()

assert args.maxsize > 10000000
assert args.maxcount < 1000000

os.makedirs(Path(args.shards), exist_ok=True)
SPLITS = ["train", "val", "test"]

tar_count = 0

with open(os.path.join(args.shards, "splits.csv"), "a+", newline="", encoding="utf-8") as f:
    csv_writer = writer(f)
    csv_writer.writerow(["tar_file", "split"])

# This is the output pattern under which we write shards.
pattern = os.path.join(args.shards, "ds_%06d.tar" + (".gz" if args.compression else ""))
with wds.ShardWriter(pattern, maxsize=int(args.maxsize), maxcount=int(args.maxcount)) as sink:
    for split in SPLITS:
        path = Path(os.path.join(args.data, split))
        text_files_l = [*path.glob("*.txt")]
        text_files = {text_file.stem: text_file for text_file in text_files_l}
        text_total = len(text_files)

        if args.json:
            json_files_l = [*path.glob("*.json")]
            json_files = {json_file.stem: json_file for json_file in json_files_l}
            json_dicts = {}

            for key in json_files:
                try:
                    with open(json_files[key], "r", encoding="utf-8") as f:
                        json_dicts[key] = json.dumps(json.load(f))
                except json.JSONDecodeError:
                    print(f"Found {len(json_files.keys()) - len(json_dicts.keys())} corrupt json file(s).")
            json_keys = json_files.keys()

        npy_files_l = [*path.glob("*.npy")]
        npy_files = {npy_file.stem: npy_file for npy_file in npy_files_l}
        npy_total = len(npy_files)

        print("Found {text_total} textfiles and {npy_total} numpy files.")
        keys = list(npy_files.keys() & text_files.keys())

        text_files = {k: v for k, v in text_files.items() if k in keys}
        npy_files = {k: v for k, v in npy_files.items() if k in keys}

        total_pairs = len(keys)
        keys = list(keys)

        split_tar_count = total_pairs // args.maxcount + (total_pairs % args.maxcount != 0)
        tar_split = [("ds_{tar_count+i:06}", split) for i in range(split_tar_count)]
        with open(os.path.join(args.shards, "splits.csv"), "a+", newline="", encoding="utf-8") as f:
            csv_writer = writer(f)
            for row in tar_split:
                csv_writer.writerow(row)
        tar_count += split_tar_count

        indexes = list(range(total_pairs))
        random.shuffle(indexes)

        for i in indexes:
            embeddings = np.load(npy_files[keys[i]])
            with open(text_files[keys[i]], "rb", encoding="utf-8") as txtstream:
                text = txtstream.read()

            ds_key = keys[i]

            sample = {"__key__": ds_key, "npy": embeddings, "cap": text}
            if args.json and keys[i] in json_keys:
                sample["json"] = json_dicts[keys[i]]
            sink.write(sample)
        sink.next_stream()
