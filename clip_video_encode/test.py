from reader import VideoReader
from framebucket import FrameBucket

# vids = ["test_data/vid1.mp4", "test_data/vid2.mp4", "https://www.youtube.com/watch?v=EKtBQbK4IX0"]
vids = ["test_data/vid1.mp4", "test_data/vid2.mp4"]

fb = FrameBucket(
  vids
)

vr = VideoReader(
  vids,
  fb,
)

vr.generate_frames()

print(len(fb.frame_dict[fb.keys[1]]))








