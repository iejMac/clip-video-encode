import torch

from multiprocessing.pool import ThreadPool


POSTPROC_SHAPE = (3, 224, 224)
DIV_COUNT = 10


class FrameBatcher:
  def __init__(self, preprocess):
    self.preprocess = preprocess

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

    torch_batch = torch.zeros((size, *POSTPROC_SHAPE))
    chunk_size = int(size/DIV_COUNT)

    with ThreadPool(DIV_COUNT) as pool:
      def prepro_samples(i):
        sl = slice(i * chunk_size, (i+1) * chunk_size) if (i < DIV_COUNT - 1)  else slice(i * chunk_size, None)
        torch_batch[sl] = torch.stack([self.preprocess(fr) for fr in batch[sl]])

      for _ in pool.imap_unordered(prepro_samples, range(DIV_COUNT)):
        pass

    return torch_batch, vid_inds
