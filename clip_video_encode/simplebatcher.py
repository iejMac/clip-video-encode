import torch


class FrameBatcher:
  def __init__(self):
    self.frame_dict = {}

  
  def __len__(self):
    l = 0
    for key in self.frame_dict.keys():
      l += len(self.frame_dict[key])
    return l

  def add_frames(self, frames, dst):
    self.frame_dict[dst] = frames

  def get_batch(self, size):
    batch = []
    for key, frames in list(self.frame_dict.items()):
      size_left = size - len(batch)
      batch += frames[:size_left]

      if size_left >= len(frames):
        self.frame_dict.pop(key)
      else:
        self.frame_dict[key] = frames[size_left:]
    return torch.stack(batch)
