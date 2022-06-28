import clip
import numpy as np
import sys
import torch

from matplotlib import pyplot as plt


def conv_filter(probs, width=10):
    padded_probs = np.pad(probs, width//2)
    prob_cp = np.zeros(probs.shape)
    for i in range(len(probs)):
      prob_cp[i] = np.mean(padded_probs[i:i+width])

    return prob_cp
      

VIDEO = "pCUtPE4cAsk.npy"
device = "cuda" if torch.cuda.is_available() else "cpu"
video_embs = torch.Tensor(np.load(VIDEO)).to(device)

chosen_thing = "grizly bear"
labels = [f"a photo of a {chosen_thing}", "a photo of something"]
tokenized_labels = clip.tokenize(labels).to(device)

model, _ = clip.load("ViT-B/32", device=device)

with torch.no_grad():
    lab_embs = model.encode_text(tokenized_labels)

    video_embs = video_embs / video_embs.norm(dim=-1, keepdim=True)
    lab_embs = lab_embs / lab_embs.norm(dim=-1, keepdim=True)

    logit_scale = model.logit_scale.exp()
    logits_per_frame = logit_scale * video_embs @ lab_embs.t()

    probs = logits_per_frame.softmax(dim=-1).cpu().numpy()


T = 12.95 # length of video in minutes
ps = probs[:, 0]
xs = [(i*T)/len(ps) for i in range(len(ps))]

plt.plot(xs, ps)
plt.show()

plt.figure().clear()

# Filter probs:
n_filter_steps = 20
for i in range(n_filter_steps):
    plt.plot(xs, ps)
    plt.savefig(f"filter_steps/filt{i}")
    plt.figure().clear()
    ps = conv_filter(ps, 20)


plt.plot(xs, ps)
plt.savefig(f"filter_steps/filt20")
plt.figure().clear()

# Interpret sorted signal:
# threshold = 0.7
# ps = [1.0 if p > threshold else 0.0 for p in ps]

plt.plot(xs, ps)
plt.show()
