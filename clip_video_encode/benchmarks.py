import glob
import numpy as np
import os
import time

from reader import VideoReader
from batcher import FrameBatcher

from torchvision.transforms import ToPILImage, Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


VID_DIR = "test_data"

def bench_reader():

  vids = glob.glob(os.path.join(VID_DIR, "*.mp4"))

  vr = VideoReader()
  fr_count = 0

  CHUNK_SIZE = len(vids) # TODO: optimize this with full test loop

  start_time = time.perf_counter()
  while len(vids) > 0:
    vid_chunk = vids[:CHUNK_SIZE]
    vids = vids[CHUNK_SIZE:]

    frames = vr.read_vids(vid_chunk)
    for k, v in frames.items():
      fr_count += len(v)

  read_time = time.perf_counter() - start_time

  perf = fr_count / read_time
  comp = (perf / 1800)

  print(f"VideoReader performance on {fr_count} frames: {perf} FPS ({comp}x of min 1800 FPS)")


def bench_batcher():
  n_px = 224
  def _convert_image_to_rgb(image):
    return image.convert("RGB")
  prepro = Compose([
    ToPILImage(),
    Resize(n_px, interpolation=BICUBIC),
    CenterCrop(n_px),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
  ])

  fb = FrameBatcher(prepro)

  # Add fake videos:
  N_VIDS = 100
  N_FRAMES = 600
  for i in range(N_VIDS):
    fb.add_frames([np.zeros((640, 480, 3)).astype(np.uint8) + i for j in range(N_FRAMES)], str(i))


  BATCH_SIZE = 200
  NUM_WORKERS = 12

  start_time = time.perf_counter()
  fr_dl, vi = fb.get_dataloader(BATCH_SIZE, NUM_WORKERS)

  for b in fr_dl:
    pass

  batch_time = time.perf_counter() - start_time
  fr_count = N_VIDS * N_FRAMES
  perf = fr_count / batch_time
  comp = (perf / 1800)

  print(f"FrameBatcher performance on {fr_count} frames: {perf} FPS ({comp}x of min 1800 FPS)")


if __name__ == "__main__":
  bench_reader()
	bench_batcher()
