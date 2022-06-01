import glob
import os
import time

from reader import VideoReader

from skvideo.io import vread

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

if __name__ == "__main__":
  bench_reader()

