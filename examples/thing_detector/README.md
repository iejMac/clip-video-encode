# Thing detector using clip-video-encode 🔍

You can use clip-video-encode to look for things in videos so let's look for a bear in an animal compilation.

## Install:
```
pip install clip-video-encode
```

## Choose video and encode (only every 10th frame, no need for more):
```
# Random animal compilation - https://www.youtube.com/watch?v=pCUtPE4cAsk
clip-video-encode https://www.youtube.com/watch?v=pCUtPE4cAsk --take_every_nth 10
```

## Load frame embeddings and tokenize text:
```python
EMBEDDINGS = "pCUtPE4cAsk.npy"
device = "cuda" if torch.cuda.is_available() else "cpu"
video_embs = torch.Tensor(np.load(EMBEDDINGS)).to(device)

chosen_thing = "bear"
labels = [f"a photo of a {chosen_thing}", "a photo of something"]
tokenized_labels = clip.tokenize(labels).to(device)
```

## Load model and get similarity scores:
```python
model, _ = clip.load("ViT-B/32", device=device)

with torch.no_grad():
    lab_embs = model.encode_text(tokenized_labels)

    video_embs = video_embs / video_embs.norm(dim=-1, keepdim=True)
    lab_embs = lab_embs / lab_embs.norm(dim=-1, keepdim=True)

    logit_scale = model.logit_scale.exp()
    logits_per_frame = logit_scale * video_embs @ lab_embs.t()

    probs = logits_per_frame.softmax(dim=-1).cpu().numpy()
```

## Let's look at a graph of the probability that the animal is present in the frame:
```python
T = 12.95 # length of video in minutes
ps = probs[:, 0]
xs = [(i*T)/len(ps) for i in range(len(ps))]

plt.plot(xs, ps)
plt.show()
```
<p align="center">
  <img src="assets/initial_prob_bear.png" height="500"/>
</p>

## Looks noisy, why don't we filter the signal:
```python
def conv_filter(probs, width=10):
    padded_probs = np.pad(probs, width//2)
    prob_cp = np.zeros(probs.shape)
    for i in range(len(probs)):
      prob_cp[i] = np.mean(padded_probs[i:i+width])
    return prob_cp

# Filter probs:
n_filter_steps = 20
for i in range(n_filter_steps):
    ps = conv_filter(ps, 20)
```

<p align="center">
  <img src="assets/filtering.gif" height="500"/>
</p>

## It looks like there might be a bear around 2:30-3:00, let's check!
<p align="center">
  <img src="assets/bear.png" height="500"/>
</p>
