"""encode video with CLIP"""
import sys

import clip
import numpy as np
import torch

from multiprocessing import SimpleQueue, Process, shared_memory
from torchvision.transforms import ToPILImage, Compose, ToTensor, Normalize

# from .reader_ffmpeg import read_vids
from .batcher import get_dl
from .reader import read_vids
from .simplemapper import FrameMapper
from .writer import write_embeddings


BATCH_SIZE = 256
VID_CHUNK_SIZE = 100
EMB_DIM = 512
QUALITY = "360p"
N_DATASET_WORKERS = 8


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

    info_q = SimpleQueue()
    complete_q = SimpleQueue()  # TODO: SharedMemory hack, do properly

    fm = FrameMapper(model)
    vr_proc = Process(target=read_vids, args=(fnames, info_q, complete_q, VID_CHUNK_SIZE, take_every_nth))
    vr_proc.start()

    while True:
        info = info_q.get()
        if isinstance(info, str):
            break

        shm = shared_memory.SharedMemory(name=info["shm_name"])
        block = np.ndarray((info["frame_count"], 224, 224, 3), dtype=np.uint8, buffer=shm.buf)
        dl = get_dl(block, preprocess, BATCH_SIZE, N_DATASET_WORKERS)

        embeddings = []
        for batch in dl:
            with torch.no_grad():
                emb = fm(batch.to(device))
                embeddings.append(emb)

        embeddings = np.concatenate(embeddings)
        write_embeddings(info["ind_dict"], embeddings, dest)
        shm.close()

    complete_q.put("DONE_MAPPING")  # TODO: SharedMemory hack, do properly
    vr_proc.join()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python clip-video-encode.py video.mp4 embeddings.npy take_every_nth")
        sys.exit(1)

    clip_video_encode(sys.argv[1], sys.argv[2], int(sys.argv[3]))
