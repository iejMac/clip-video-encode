
from bucket import FrameBucket

import time
from copy import deepcopy


class FrameBatcher:
  """Watches FrameBucket and extracts full batches to send to mapper"""
  
  def __init__(
    self,
    frame_bucket,
    batch_size=256,
  ):
    """
      Input:
        frame_bucket: FrameBucket object to watch for full batches
        batch_size: max size of batch to send to mapper
    """
    self.framebucket = frame_bucket
    self.batch_size = batch_size

    # self.not_done = deepcopy(self.framebucket.keys)

  def make_batches(self):
    while True:
      '''
      for key in self.not_done:
        if self.framebucket.dst_dict[key] != None:
          print("============")
          print(f"{key} done!")
          print(f"{key} has {len(self.framebucket.frame_dict[key])} frames of shape {self.framebucket.frame_dict[key][0].shape}")
          self.not_done.remove(key)
          self.framebucket.frame_dict[key] = [] # empty video frames
      '''

      if len(self.framebucket) >= self.batch_size:
        batch = []
        for key in self.framebucket.keys:
          cur_ind = self.framebucket.used_ind[key]

          frame_count = len(self.framebucket.frame_dict[key]) - cur_ind # n frames unused
          batch_left = batch_size - len(batch)
          
          use_from_key = number
          self.framebucket.used_ind[key] += use_from_key
          
          batch += self.framebucket.frame_dict[key][cur_ind:cur_ind + use_from_key]
        

      if len(self.not_done) == 0:
        break
      time.sleep(0.01)
      
