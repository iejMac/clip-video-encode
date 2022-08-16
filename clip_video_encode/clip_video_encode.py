"""encode video with CLIP"""
import sys

import clip
import numpy as np
import torch

from torchvision.transforms import ToPILImage, Compose, ToTensor, Normalize
from video2numpy.frame_reader import FrameReader

from .reader import Reader
from .simplemapper import FrameMapper
from .utils import block2dl
from .writer import FileWriter, WebDatasetWriter


BATCH_SIZE = 256
IMG_SIZE = 224
EMB_DIM = 512
N_DATASET_WORKERS = 6
CHUNK_SIZE = 200


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def clip_video_encode(
    src,
    dest="",
    output_format="files",
    take_every_nth=1,
    frame_workers=1,
    frame_memory_size=4,
    metadata_columns="",
    use_dst_name=False,
):
    """
    Encode frames using CLIP image encoder

    Input:
      src:
        str: path to mp4 file
        str: youtube link
        str: path to txt file with multiple mp4's or youtube links
        list: list with multiple mp4's or youtube links
      dest:
        str: directory where to save embeddings to
        None: dest = src + .npy
      output_format:
        str: "files" or "webdataset"
      take_every_nth:
        int: only take every nth frame
      frame_workers:
        int: number of Processes to distribute video reading to.
      frame_memory_size:
        int: GB of memory for FrameReader.
      metadata_columns:
        str: a comma separated list of metadata column names to look for in src
      use_dst_name:
        bool: use the save name suggested by video2numpy
    """
    if isinstance(metadata_columns, str):
        metadata_columns = [metadata_columns] if metadata_columns != "" else []
    metadata_columns = list(metadata_columns) if isinstance(metadata_columns, tuple) else metadata_columns
    reader = Reader(src, metadata_columns)
    vids, ids, meta = reader.get_data()
    meta_refs = list(range(len(vids)))

    assert output_format in ["files", "webdataset"]
    if output_format == "files":
        writer = FileWriter(dest)
    elif output_format == "webdataset":
        # TODO: maybe include params for this?
        writer = WebDatasetWriter(dest, 9, "npy", maxcount=10000, shard_id=0)

    # Initialize model:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    preprocess = Compose(
        [
            ToPILImage(),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )

    fm = FrameMapper(model, device)
    fr = FrameReader(vids, meta_refs, take_every_nth, IMG_SIZE, workers=frame_workers, memory_size=frame_memory_size)
    fr.start_reading()

    frames, ind_dict = [], {}
    i = 1
    for vid_frames, info in fr:
        frames.append(vid_frames)
        ind_dict[info["reference"]] = (len(frames), len(frames) + vid_frames.shape[0], info["dst_name"])

        if (i % CHUNK_SIZE == 0) or (i == len(fr)):
            vid_block = np.concatenate(frames)
            dl = block2dl(vid_block, preprocess, BATCH_SIZE, 12)

            embeddings = []
            for batch in dl:
                with torch.no_grad():
                    emb = fm(batch.to(device))
                    embeddings.append(emb)

            embeddings = np.concatenate(embeddings)
            for ref, (i0, it, dst_name) in ind_dict.items():
                vid_id = dst_name[:-4] if use_dst_name else ids[ref]
                vid_meta = {}
                for k in meta:
                    vid_meta[k] = meta[k][ref].as_py()
                writer.write(embeddings[i0:it], vid_id, vid_meta)
            frames, ind_dict = [], {}
        i += 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python clip-video-encode.py video.mp4 embeddings.npy take_every_nth")
        sys.exit(1)

    clip_video_encode(sys.argv[1], sys.argv[2], int(sys.argv[3]))
