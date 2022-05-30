import time

import clip
import torch

from torchvision.transforms import Compose, ToPILImage

from simplereader import VideoReader
from simplebatcher import FrameBatcher
from simplemapper import FrameMapper
from simplewriter import EmbeddingWriter



BATCH_SIZE = 100


vids = ["test_data/vid1.mp4", "test_data/vid2.mp4", "test_data/vid3.mp4", "test_data/vid4.mp4"]


reader_vids = vids # later this might be only a subset of vids and multiple readers


# Initialize model:
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
preprocess = Compose([ToPILImage(), preprocess])


vr = VideoReader(preprocess=preprocess)
fb = FrameBatcher()
fm = FrameMapper(model=model)
ew = EmbeddingWriter(destination_dir="test_npy")

vid_start_time = time.perf_counter()

for vid in vids:

  start_time = time.perf_counter()
  frames, dst_name = vr.generate_frames(vid)
  read_time = time.perf_counter() - start_time
  print(f"Read {len(frames)} frames time = {read_time}")

  fb.add_frames(frames, dst_name)
  ew.init_video(dst_name, len(frames))

  while len(fb) >= BATCH_SIZE:
    start_time = time.perf_counter()
    batch, vid_inds = fb.get_batch(BATCH_SIZE)
    batch = batch.to(device)
    print(vid_inds)
    batch_time = time.perf_counter() - start_time
    print(f"Batch {batch.shape} time = {batch_time}")

    start_time = time.perf_counter()
    embeddings = fm(batch)
    model_time = time.perf_counter() - start_time
    print(f"Model {embeddings.shape} time = {model_time}")

    # Separate by video
    for v, i0, it in vid_inds:
      vid_embeddings = ew.add_embeddings(v, embeddings[i0:it])

  start_time = time.perf_counter()
  flushed_count = ew.flush() 
  if flushed_count > 0:
    flush_time = time.perf_counter() - start_time
    print(f"Flushed {flushed_count} embeddings time = {flush_time}")


# Get leftover batches:
'''
print("LEFTOVER BATCHES:")
JUST COPY FROM MAIN LOOP ONCE DONE WITH get_batch(len(fb))
'''

import numpy as np
for k, v in ew.embeddings.items():
  print(np.concatenate(v).shape)


proc_vid_time = time.perf_counter() - vid_start_time
print(f"Time to process all vids = {proc_vid_time}")


