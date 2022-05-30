

class FrameBucket:
  """Data structure that stores frames from VideoReaders"""

  def __init__(self, videos):
    """
    Input:
      videos: list of videos (path or link) to store
    """
    self.keys = videos

    self.frame_dict = dict([(vid, []) for vid in videos]) # vid, frames
    self.dst_dict = dict([(vid, None) for vid in videos]) # vid, dst (filled out by VideoReader)

    self.used_ind = dict([(vid, 0) for vid in videos])
    
  def __len__(self):
    l = 0
    for key in self.keys:
      l += len(self.frame_dict[key])
    return l

  def add_frame(self, vid, frame):
    self.frame_dict[vid].append(frame)
   
  def set_dst(self, vid, dst):
    self.dst_dict[vid] = dst
