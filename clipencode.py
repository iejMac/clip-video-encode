import clip
import numpy as np
import skvideo.io as skv
import torch

from PIL import Image
from torchvision.transforms import Compose, ToPILImage, ToTensor


#TODO: make this work with any encoder model
def clipencode(src, dest):
  # Initialize model:
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = clip.load("ViT-B/32", device=device)
  preprocess = Compose([ToPILImage(), preprocess])

  # vid = np.moveaxis(skv.vread(path), -1, 1)
  vid = skv.vread(path)
  video_embeddings = np.zeros((len(vid), 512)) # TODO: get embed size from model

  with torch.no_grad():
    for i, frame in enumerate(vid):
      img = preprocess(frame).unsqueeze(0).to(device)
      video_embeddings[i] = model.encode_image(img).cpu().detach().numpy()

  np.save(dest, video_embeddings)


path = "video.mp4"
device = "cuda" if torch.cuda.is_available() else "cpu"
# clipencode(path, "video.npy")

video_embs = torch.Tensor(np.load("video.npy")).to(device)

# labels = ["a basketball hoop", "a basketball going into a basketball hoop", "a basketball missing a basketball hoop"]
labels = ["basketball", "dog", "cat"]

test = True
if test:
  model, preprocess = clip.load("ViT-B/32", device=device)

  text = clip.tokenize(labels).to(device)

  with torch.no_grad():
    text_embs = model.encode_text(text)

    video_embs = video_embs / video_embs.norm(dim=1, keepdim=True)
    text_embs = text_embs / text_embs.norm(dim=1, keepdim=True)

    logit_scale = model.logit_scale.exp()
    logits_per_frame = logit_scale * video_embs @ text_embs.t()
    
    probs = logits_per_frame.softmax(dim=-1).cpu().numpy()

  print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
