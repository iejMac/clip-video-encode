"""encode video with CLIP"""
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


def clip_video_encode(src, dest):
    """
    Encode frames using CLIP image encoder

    Input:
      src: path to mp4 file
      output: where to save embeddings.npy
    Output:
      None
    """
    if src.endswith(".mp4"):  # mp4 file
        fname = src
    else:  # youtube link
        ydl_opts = {}
        ydl = youtube_dl.YoutubeDL(ydl_opts)
        formats = ydl.extract_info(src, download=False).get("formats", None)
        f = None
        for f in formats:
            if f.get("format_note", None) != QUALITY:
                continue
            break
        fname = f.get("url", None)

    cap = cv2.VideoCapture(fname)  # pylint: disable=I1101
    if not cap.isOpened():
        print("Error: Video not opened")
        sys.exit(1)

    # Initialize model:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    preprocess = Compose([ToPILImage(), preprocess])

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
    np.save(dest, video_embeddings)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clip-video-encode.py video.mp4 embeddings.npy")
        sys.exit(1)

    clip_video_encode(sys.argv[1], sys.argv[2])
