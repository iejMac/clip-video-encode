import os
import glob
import clip
import torch
import time

from torchvision.transforms import ToPILImage, Compose, ToTensor, Normalize

from clip_video_encode.live_numpy_encoder import LiveNumpyEncoder
from clip_video_encode.simplemapper import FrameMapper


def _convert_image_to_rgb(image):
    return image.convert("RGB")


DATA_DIR = "nps" # load up DATA_DIR with numpy video frame arrays (https://github.com/iejMac/video2numpy)
                 # you can do this live while LiveNumpyEncoder is functioning as long as you pass it
                 # the entire set of fnames you expect encoded.

EMB_DIR = "embs" # save embeddings here

VIDS = os.listdir(DATA_DIR)

print(f"DATA_DIR has {len(os.listdir(DATA_DIR))} frame arrays")
print(f"EMB_DIR has {len(os.listdir(EMB_DIR))} embedding arrays")

# Initialize model and preproc:
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

np_enc = LiveNumpyEncoder(DATA_DIR, EMB_DIR, VIDS, fm, preprocess)
np_enc.start()

print("DONE ENCODING")
print(f"DATA_DIR has {len(os.listdir(DATA_DIR))} frame arrays")
print(f"EMB_DIR has {len(os.listdir(EMB_DIR))} embedding arrays")
# Once this finishes EMB_DIR should have a embedding array for each vid in VIDS
