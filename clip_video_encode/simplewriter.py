import os

import numpy as np

class EmbeddingWriter:
  def __init__(self, destination_dir):
    self.dest_dir = destination_dir

    self.embeddings = {}
    self.frame_counts = {}
 
  def init_video(self, dst, num_frames):
    self.embeddings[dst] = []
    self.frame_counts[dst] = num_frames

  def add_embeddings(self, dst, embeddings):
    self.embeddings[dst].append(embeddings)
    self.frame_counts[dst] -= len(embeddings)


  def flush(self):
    """Write video embeddings that are ready to be written"""
    flushed_count = 0
    for dst, count in list(self.frame_counts.items()):
      if count == 0:
        full_embed = np.concatenate(self.embeddings[dst])
        flushed_count += len(full_embed)
        save_pth = os.path.join(self.dest_dir, dst)
        np.save(save_pth, full_embed)
        
        # Remove since no longer needed
        self.embeddings.pop(dst) 
        self.frame_counts.pop(dst)

    return flushed_count
