"""encode video with CLIP"""
import os
import sys

import clip
import cv2
import numpy as np
import torch
import youtube_dl

from torchvision.transforms import Compose, ToPILImage


BS = 256
EMB_DIM = 512
QUALITY = "360p"


def clip_video_encode(src, dest=None):
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
        None: dst = src + .npy

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
    model, preprocess = clip.load("ViT-B/32", device=device)
    preprocess = Compose([ToPILImage(), preprocess])

    for fname in fnames:
        if not fname.endswith(".mp4"):  # youtube link
            ydl_opts = {}
            ydl = youtube_dl.YoutubeDL(ydl_opts)
            info = ydl.extract_info(fname, download=False)
            formats = info.get("formats", None)
            f = None
            for f in formats:
                if f.get("format_note", None) != QUALITY:
                    continue
                break

            fname = f.get("url", None)

            dst_name = info.get("id") + ".npy"
            dst = dst_name if dest is None else os.path.join(dest, dst_name)
        else:
            dst_name = fname[:-4].split("/")[-1] + ".npy"
            dst = fname[:-4] + ".npy" if dest is None else os.path.join(dest, dst_name)

        cap = cv2.VideoCapture(fname)  # pylint: disable=I1101
        if not cap.isOpened():
            print("Error: Video not opened")
            sys.exit(1)

        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # pylint: disable=I1101
        video_embeddings = np.zeros((fc, EMB_DIM))
        batch = []

        ret = True
        counter = 0
        while ret:
            ret, frame = cap.read()

            if (len(batch) == BS) or ((not ret) and (len(batch) > 0)):  # encode
                t_batch = torch.stack(batch).to(device)
                video_embeddings[counter : counter + len(batch)] = model.encode_image(t_batch).cpu().detach().numpy()
                counter += len(batch)
                batch = []

            if ret:
                batch.append(preprocess(frame))

        video_embeddings = video_embeddings[:counter]
        np.save(dst, video_embeddings)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clip-video-encode.py video.mp4 embeddings.npy")
        sys.exit(1)

    clip_video_encode(sys.argv[1], sys.argv[2])
