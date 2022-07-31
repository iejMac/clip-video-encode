"""encode video with CLIP"""
import os
import sys

import clip
import numpy as np
import torch

from torchvision.transforms import ToPILImage, Compose, ToTensor, Normalize
from video2numpy.frame_reader import FrameReader

from .simplemapper import FrameMapper

# from .writer import write_embeddings
from .utils import block2dl


BATCH_SIZE = 256
IMG_SIZE = 224
EMB_DIM = 512
N_DATASET_WORKERS = 6


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def clip_video_encode(src, dest="", take_every_nth=1):
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
      take_every_nth:
        int: only take every nth frame

    Output:
      None
    """
    if isinstance(src, str):
        if src.endswith(".txt"):  # list of mp4s or youtube links
            with open(src, "r", encoding="utf-8") as f:
                fnames = [fn[:-1] for fn in f.readlines()]
        else:  # mp4 or youtube link
            fnames = [src]
    else:
        fnames = src

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
    fr = FrameReader(fnames, take_every_nth, IMG_SIZE, workers=N_DATASET_WORKERS)
    fr.start_reading()

    for vid_block, info in fr:
        dl = block2dl(vid_block, preprocess, BATCH_SIZE, 0)

        embeddings = []
        for batch in dl:
            with torch.no_grad():
                emb = fm(batch.to(device))
                embeddings.append(emb)

        embeddings = np.concatenate(embeddings)
        save_pth = os.path.join(dest, info["dst_name"])
        np.save(save_pth, embeddings)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python clip-video-encode.py video.mp4 embeddings.npy take_every_nth")
        sys.exit(1)

    clip_video_encode(sys.argv[1], sys.argv[2], int(sys.argv[3]))
