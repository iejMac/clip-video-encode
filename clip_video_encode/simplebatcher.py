import torch


class FrameBatcher:
  def __init__(self):
    self.frames = {}
  
  def __len__(self):
    l = 0
    for vid in self.frames.keys():
      l += len(self.frames[vid])
    return l

  def add_frames(self, frames, dst):
    self.frames[dst] = frames

  def get_batch(self, size):
    batch = []
    vid_inds = []
    for vid, frames in list(self.frames.items()):
      # inds = [vid, start_ind_of_vid, end_ind_of_vid]
      inds = [vid, len(batch), -1]

      size_left = size - len(batch)
      batch += frames[:size_left]
      inds[2] = len(batch)

      vid_inds.append(inds)
      if size_left >= len(frames):
        self.frames.pop(vid)
      else:
        self.frames[vid] = frames[size_left:]
    return torch.stack(batch), vid_inds
