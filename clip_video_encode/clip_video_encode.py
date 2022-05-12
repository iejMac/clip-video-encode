import clip
import numpy as np
import skvideo.io as skv
import sys
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage


BS = 256

#TODO: make this work with any encoder model
def clip_video_encode(src, dest):
  # Initialize model:
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = clip.load("ViT-B/32", device=device)
  preprocess = Compose([ToPILImage(), preprocess])

  vid = skv.vread(src)
  video_embeddings = np.zeros((len(vid), 512)) # TODO: get embed size from model

  proc_video = torch.zeros((len(vid), 3, 224, 224))
  for i, frame in enumerate(vid):
    proc_video[i] = preprocess(frame).unsqueeze(0)

  # Batch up and embed video
  ds_vid = DataLoader(proc_video, batch_size=BS)
  for i, b in enumerate(ds_vid):
    imgs = b.to(device)
    video_embeddings[i*BS:(i+1)*BS] = model.encode_image(imgs).cpu().detach().numpy()

  np.save(dest, video_embeddings)

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python clip-video-encode.py video.mp4 embeddings.npy")
    sys.exit(1)

  clip_video_encode(sys.argv[1], sys.argv[2])
