import time

import clip
import torch

from torchvision.transforms import Compose, ToPILImage

from reader import VideoReader
from simplebatcher import FrameBatcher
from simplemapper import FrameMapper
from simplewriter import EmbeddingWriter


BATCH_SIZE = 100
CHUNK_SIZE = 100


vids = ["test_data/vid1.mp4", "test_data/vid2.mp4", "test_data/vid3.mp4", "test_data/vid4.mp4"]


reader_vids = vids # later this might be only a subset of vids and multiple readers

if __name__ == "__main__":
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

    while len(fb) >= BATCH_SIZE:
      start_time = time.perf_counter()
      batch, vid_inds = fb.get_batch(BATCH_SIZE)
      batch = batch.to(device)
      batch_time = time.perf_counter() - start_time
      print(f"Batch rate {batch.shape[0] / batch_time} [samples/s]")

      start_time = time.perf_counter()
      embeddings = fm(batch)
      model_time = time.perf_counter() - start_time
      print(f"Model rate {embeddings.shape[0] / model_time} [samples/s]")

      # Separate by video
      for v, i0, it in vid_inds:
        vid_embeddings = ew.add_embeddings(v, embeddings[i0:it])

    start_time = time.perf_counter()
    flushed_count = ew.flush() 
    if flushed_count > 0:
      flush_time = time.perf_counter() - start_time
      print(f"Flushed rate {flushed_count / flush_time} [samples/s]")


  # Get leftover batches:
  '''
  print("LEFTOVER BATCHES:")
  JUST COPY FROM MAIN LOOP ONCE DONE WITH get_batch(len(fb))
  import numpy as np
  for k, v in ew.embeddings.items():
    print(np.concatenate(v).shape)
  '''

  proc_vid_time = time.perf_counter() - vid_start_time
  print(f"TOTAL FRAME RATE: {TOT_FR_COUNT/proc_vid_time} [samples/s]")

