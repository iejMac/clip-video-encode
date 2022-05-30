from reader import VideoReader
from bucket import FrameBucket
from batcher import FrameBatcher

from threading import Thread

# vids = ["test_data/vid1.mp4", "test_data/vid2.mp4", "test_data/vid3.mp4", "test_data/vid4.mp4", "https://www.youtube.com/watch?v=EKtBQbK4IX0"]
vids = ["test_data/vid1.mp4", "test_data/vid2.mp4", "test_data/vid3.mp4", "test_data/vid4.mp4"]

fb = FrameBucket(
  vids,
)

reader_vids = vids # later this might be only a subset of vids and multiple readers

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




