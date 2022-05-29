

class FrameBucket:
  """Data structure that stores frames from VideoReaders"""

  def __init__(self, videos):
    """
    Initializes a FrameBucket object

    Input:
      videos: list of videos (path or link) to store
    """
    self.keys = videos

    self.frame_dict = dict([(vid, []) for vid in videos]) # vid, frames
    self.dst_dict = dict([(vid, None) for vid in videos]) # vid, dst (filled out by VideoReader)
    

  def add_frame(self, vid, frame):
    self.frame_dict[vid].append(frame)
   
  def set_dst(self, vid, dst):
    self.dst_dict[vid] = dst

