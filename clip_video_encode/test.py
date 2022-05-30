import time


from torchvision.transforms import ToTensor

from simplereader import VideoReader
from simplebatcher import FrameBatcher



BATCH_SIZE = 60


# vids = ["test_data/vid1.mp4", "test_data/vid2.mp4", "test_data/vid3.mp4", "test_data/vid4.mp4", "https://www.youtube.com/watch?v=EKtBQbK4IX0"]
vids = ["test_data/vid1.mp4", "test_data/vid2.mp4", "test_data/vid3.mp4", "test_data/vid4.mp4"]


reader_vids = vids # later this might be only a subset of vids and multiple readers

prepro = ToTensor()


vr = VideoReader(preprocess=prepro)
fb = FrameBatcher()


vid_start_time = time.perf_counter()

for vid in vids:

  start_time = time.perf_counter()
  frames, dst_name = vr.generate_frames(vid)
  read_time = time.perf_counter() - start_time
  print(f"Read {len(frames)} frames time = {read_time}")

  fb.add_frames(frames, dst_name)

  while len(fb) >= BATCH_SIZE:
    start_time = time.perf_counter()
    batch = fb.get_batch(BATCH_SIZE)
    batch_time = time.perf_counter() - start_time

    print(f"Batch {batch.shape} tensor time = {batch_time}")



# Get leftover batches:
print("LEFTOVER BATCHES:")
while len(fb) > 0:
  start_time = time.perf_counter()
  batch = fb.get_batch(BATCH_SIZE)
  batch_time = time.perf_counter() - start_time

  print(f"Batch {batch.shape} tensor time = {batch_time}")


proc_vid_time = time.perf_counter() - vid_start_time
print(f"Time to process all vids = {proc_vid_time}")



''' Multithreading start, paused for now
from reader import VideoReader
from bucket import FrameBucket
from batcher import FrameBatcher

from threading import Thread

fb = FrameBucket(
  vids,
)

vr = VideoReader(
  reader_vids,
  fb,
)
bt = FrameBatcher(
  fb,
  20,
)

# Fill up frame bucket with frames
thr1 = Thread(target=vr.generate_frames)
thr2 = Thread(target=bt.make_batches)

thr1.start()
thr2.start()



thr1.join()
thr2.join()

# print(fb.frame_dict)
'''
