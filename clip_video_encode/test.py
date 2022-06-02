import os
import time

import glob
import clip
import torch
import numpy as np

from torchvision.transforms import Compose, ToPILImage

from reader import VideoReader
from batcher import FrameBatcher
from simplemapper import FrameMapper
from simplewriter import EmbeddingWriter

BATCH_SIZE = 256
NUM_WORKERS = 12
CHUNK_SIZE = 100

VID_DIR = "../../wds_kinetics/cve_tests/big_test_vids/"
vids = glob.glob(os.path.join(VID_DIR, "*.mp4"))
vids = vids[:200]
# vids = [f"test_data/vid{i + 1}.mp4" for i in range(10)]


reader_vids = vids # later this might be only a subset of vids and multiple readers


# Initialize model:
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
preprocess = Compose([ToPILImage(), preprocess])


vr = VideoReader()
fb = FrameBatcher(preprocess=preprocess)
fm = FrameMapper(model=model)
ew = EmbeddingWriter(destination_dir="test_npy")

vid_start_time = time.perf_counter()

TOT_FR_COUNT = 0

while len(vids) > 0:
  vid_chunk = vids[:CHUNK_SIZE]
  vids = vids[CHUNK_SIZE:]

  start_time = time.perf_counter()

  frames = vr.read_vids(vid_chunk)
  fr_count = 0

  for dst, frs in frames.items():
    fb.add_frames(frs, dst)
    ew.init_video(dst, len(frs))
    fr_count += len(frs)
    TOT_FR_COUNT += len(frs)

  read_time = time.perf_counter() - start_time
  print(f"Read rate : {fr_count / read_time} [samples/s]")

  start_time = time.perf_counter()
  dl, vid_inds = fb.get_dataloader(BATCH_SIZE, NUM_WORKERS)

  embs = []
  mod_time = 0
  ct = 0
  for batch in dl:
    ct += batch.shape[0]
    mod_start = time.perf_counter()
    embs.append(fm(batch.to(device)))
    mod_time += time.perf_counter() - mod_start

  preproc_model_time = time.perf_counter() - start_time
  preproc_time = preproc_model_time - mod_time
  print(f"Preprocess rate : {ct/preproc_time} [samples/s]")
  print(f"Model rate : {ct/mod_time} [samples/s]")
  print(f"Preproc + Model rate : { ct/preproc_model_time } [samples/s]")

  embeddings = np.concatenate(embs)
  # Separate by video
  for v, i0, it in vid_inds:
    vid_embeddings = ew.add_embeddings(v, embeddings[i0:it])

  start_time = time.perf_counter()
  flushed_count = ew.flush()
  if flushed_count > 0:
    flush_time = time.perf_counter() - start_time
    print(f"Flushed rate {flushed_count / flush_time} [samples/s]")

proc_vid_time = time.perf_counter() - vid_start_time
print(f"TOTAL FRAME RATE: {TOT_FR_COUNT/proc_vid_time} [samples/s]")
print(f"TOTAL TIME: {proc_vid_time}")
