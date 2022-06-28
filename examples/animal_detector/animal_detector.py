import clip
import numpy as np
import sys
import torch

import skvideo.io as skv
from matplotlib import pyplot as plt


def conv_filter(probs, width=10):
    pad_ps = (width // 2) * [0.0] + probs + (width // 2) * [0.0]
    ret_ps = []
    for i in range(len(probs)):
        ret_ps.append(np.mean(probs[i : i + width]))
    return ret_ps


VIDEO = "examples/animals/animals"
device = "cuda" if torch.cuda.is_available() else "cpu"
video_embs = torch.Tensor(np.load(VIDEO + ".npy")).to(device)

model, preprocess = clip.load("ViT-B/32", device=device)

# animal = "bear"
# labels = [f"a {animal}", f"something that isn't a {animal}"]
labels = ["grizzly bear", "an animal that is not a grizzly bear"]
text = clip.tokenize(labels).to(device)

with torch.no_grad():
    text_embs = model.encode_text(text)

    video_embs = video_embs / video_embs.norm(dim=1, keepdim=True)
    text_embs = text_embs / text_embs.norm(dim=1, keepdim=True)

    logit_scale = model.logit_scale.exp()
    logits_per_frame = logit_scale * video_embs @ text_embs.t()

    probs = logits_per_frame.softmax(dim=-1).cpu().numpy()

T = 8 * 60 + 10
ps = []
x = []

for i, p in enumerate(probs):
    ps.append(p[0])
    x.append(i * T / 60)

plt.plot(ps)
plt.show()

# Filter probs:
n_filter_steps = 20
for i in range(n_filter_steps):
    ps = conv_filter(ps, 20)

# Interpret sorted signal:
threshold = 0.7
ps = [1.0 if p > threshold else 0.0 for p in ps]

plt.plot(ps)
plt.show()
