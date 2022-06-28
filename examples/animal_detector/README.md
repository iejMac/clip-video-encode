# Animal detector using clip-video-encode 🔍🦁

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

chosen_animal = "bear"
labels = [f"a photo of a {chosen_animal}", "a photo of an animal"]
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
T = 13 # length of video in minutes
ps = probs[:, 0].tolist()
xs = [(i*T)/len(ps) for i in range(len(ps))]

plt.plot(xs, ps)
plt.show()
```

![Initial probability of bear](assets/initial_prob_bear.png =300x)
