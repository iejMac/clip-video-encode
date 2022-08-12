"""encode video with CLIP"""
import os
import sys
import time

import clip
import fsspec
import numpy as np
import torch

from io import BytesIO
from torchvision.transforms import ToPILImage, Compose, ToTensor, Normalize
from video2numpy.frame_reader import FrameReader

from .simplemapper import FrameMapper

from .utils import block2dl


BATCH_SIZE = 256
IMG_SIZE = 224
EMB_DIM = 512
N_DATASET_WORKERS = 6
CHUNK_SIZE = 200


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def clip_video_encode(src, dest="", take_every_nth=1, frame_workers=1):
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
      frame_workers:
        int: number of Processes to distribute video reading to.
    """
    if isinstance(src, str):
        if src.endswith(".txt"):  # list of mp4s or youtube links
            with open(src, "r", encoding="utf-8") as f:
                fnames = [fn[:-1] for fn in f.readlines()]
        else:  # mp4 or youtube link
            fnames = [src]
    else:
        fnames = src

    fs, dest = fsspec.core.url_to_fs(dest)

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
    fr = FrameReader(fnames, take_every_nth, IMG_SIZE, workers=frame_workers)
    fr.start_reading()

    t0 = time.perf_counter()
    ct = 0

    frames, ind_dict = [], {}
    i = 1
    for vid_frames, info in fr:
        frames.append(vid_frames)
        ind_dict[info["dst_name"]] = (len(frames), len(frames) + vid_frames.shape[0])

        if (i % CHUNK_SIZE == 0) or (i == len(fr)):
            vid_block = np.concatenate(frames)
            dl = block2dl(vid_block, preprocess, BATCH_SIZE, N_DATASET_WORKERS)

            embeddings = []
            for batch in dl:
                with torch.no_grad():
                    emb = fm(batch.to(device))
                    embeddings.append(emb)

            embeddings = np.concatenate(embeddings)
            ct += embeddings.shape[0]
            for dst_name, (i0, it) in ind_dict.items():
                save_pth = os.path.join(dest, dst_name)
                with fs.open(save_pth, "wb") as f:
                    nbp = BytesIO()
                    np.save(nbp, embeddings[i0:it])
                    f.write(nbp.getbuffer())
            frames, ind_dict = [], {}
        i += 1

    tf = time.perf_counter()
    print(ct)
    print(tf-t0)
    print(f"FPS: {ct/(tf-t0)}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python clip-video-encode.py video.mp4 embeddings.npy take_every_nth")
        sys.exit(1)

    clip_video_encode(sys.argv[1], sys.argv[2], int(sys.argv[3]))
