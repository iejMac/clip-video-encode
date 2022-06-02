import torch

from torch.utils.data import Dataset, DataLoader

POSTPROC_SHAPE = (3, 224, 224)

class HelperDataset(Dataset):
  def __init__(
    self,
    imgs,
    preprocess,
  ):
    super().__init__()
    self.imgs = imgs
    self.preprocess = preprocess
  def __len__(self):
    return len(self.imgs)
  def __getitem__(self, ind):
    return self.preprocess(self.imgs[ind])

def ds_to_dl(ds, bs, n_work):
  return DataLoader(
    ds,
    batch_size = bs,
    shuffle = False,
    num_workers = n_work,
  )

class FrameBatcher(Dataset):
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

  def get_dataloader(self, batch_size, n_work):
    all_frames = []
    vid_inds = []
    for vid, frames in list(self.frames.items()):
      # inds = [vid, start_ind_of_vid, end_ind_of_vid]
      inds = [vid, len(all_frames), -1]
      all_frames += frames
      inds[2] = len(all_frames)

      vid_inds.append(inds)
      self.frames.pop(vid)

    ds = HelperDataset(all_frames, self.preprocess)
    dl = ds_to_dl(ds, batch_size, n_work)

    return dl, vid_inds
