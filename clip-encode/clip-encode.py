import clip
import numpy as np
import skvideo.io as skv
import sys
import torch

from torchvision.transforms import Compose, ToPILImage


#TODO: make this work with any encoder model
def clipencode(src, dest):
  # Initialize model:
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = clip.load("ViT-B/32", device=device)
  preprocess = Compose([ToPILImage(), preprocess])

  vid = skv.vread(src)
  video_embeddings = np.zeros((len(vid), 512)) # TODO: get embed size from model

  with torch.no_grad():
    for i, frame in enumerate(vid):
      img = preprocess(frame).unsqueeze(0).to(device)
      video_embeddings[i] = model.encode_image(img).cpu().detach().numpy()

  np.save(dest, video_embeddings)


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python clipencode.py video.mp4 embeddings.npy")
    sys.exit(1)

  clipencode(sys.argv[1], sys.argv[2])
