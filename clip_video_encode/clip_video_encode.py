"""encode video with CLIP"""
import os
import sys

import clip
import cv2
import numpy as np
import torch
import youtube_dl

from torchvision.transforms import Compose, ToPILImage
from tqdm import tqdm


BS = 20
EMB_DIM = 512
QUALITY = "360p"


class FrameEmbeddings:
  def __init__(self, batch_size):
    self.batch_size = batch_size

    self.frames = {}
    self.dests = {}
    self.embeddings = {}

  def _frame_count(self):
    n_frames = 0
    for arr in self.frames.values():
      n_frames += len(arr)
    return n_frames

  def batch_ready(self):
    return self._frame_count() >= self.batch_size

  def get_batch(self):
    batch = [] # frames
    inds = [] # vid, first_frame_ind, last_frame_ind
    for k, v in list(self.frames.items()):
      i_t = [k, len(batch), 0]

      if len(batch) == self.batch_size:
        break
      max_take = self.batch_size - len(batch)
      batch += v[:max_take]

      i_t[2] = len(batch)
      inds.append(i_t)
      #                         video might not be complete 
      if max_take >= len(v) and (len(self.frames.keys()) > 1): # we've taken all frames from v
        self.frames.pop(k)
      else: # only remove used frames
        self.frames[k] = v[max_take:]

    return torch.stack(batch), inds
  
  def save_vids(self, vids):
    for vid in vids: 
      dst = self.dests[vid]
      full_embeds = np.concatenate(self.embeddings[vid])
      np.save(dst, full_embeds)


def clip_video_encode(src, dest=None, take_every_nth=1):
    """
    Encode frames using CLIP image encoder

    Input:
      src:
        str: path to mp4 file
        str: youtube link
        str: path to txt file with multiple mp4's or youtube links
        list: list with multiple mp4's or youtube links
      dest:
        str: directory where to save embeddings to
        None: dst = src + .npy
      take_every_nth:
        int: only take every nth frame

    Output:
      None
    """
    if isinstance(src, str):
        if src.endswith(".txt"):  # list of mp4s or youtube links
            with open(src, "r", encoding="utf-8") as f:
                fnames = [fn[:-1] for fn in f.readlines()]
        else:  # mp4 or youtube link
            fnames = [src]
    else:
        fnames = src

    # Initialize model:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    preprocess = Compose([ToPILImage(), preprocess])

    # batch = {}
    # embeddings = {}
    frame_embeds = FrameEmbeddings(BS)

    for fname in tqdm(fnames):
        if not fname.endswith(".mp4"):  # youtube link
            ydl_opts = {}
            ydl = youtube_dl.YoutubeDL(ydl_opts)
            info = ydl.extract_info(fname, download=False)
            formats = info.get("formats", None)
            f = None
            for f in formats:
                if f.get("format_note", None) != QUALITY:
                    continue
                break

            fname = f.get("url", None)

            dst_name = info.get("id") + ".npy"
            dst = dst_name if dest is None else os.path.join(dest, dst_name)
        else:
            dst_name = fname[:-4].split("/")[-1] + ".npy"
            dst = fname[:-4] + ".npy" if dest is None else os.path.join(dest, dst_name)

        frame_embeds.frames[fname] = [] # initialize new video

        frame_embeds.dests[fname] = dst

        cap = cv2.VideoCapture(fname)  # pylint: disable=I1101
        if not cap.isOpened():
            print("Error: Video not opened")
            sys.exit(1)

        frame_embeds.embeddings[fname] = []

        # fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # pylint: disable=I1101
        # video_embeddings = np.zeros((fc, EMB_DIM))
        # batch = []

        ret = True
        # counter = 0
        ind = 0
        while ret:
            ret, frame = cap.read()
          
            '''
            if (len(batch) == BS) or ((not ret) and (len(batch) > 0)):  # encode
                t_batch = torch.stack(batch).to(device)
                video_embeddings[counter : counter + len(batch)] = model.encode_image(t_batch).cpu().detach().numpy()
                counter += len(batch)
                batch = []
            '''
            if frame_embeds.batch_ready(): # embed a full batch
                batch, inds = frame_embeds.get_batch()
                batch = batch.to(device)
                embeds = model.encode_image(batch).cpu().detach().numpy()
                for vid, i0, it  in inds:
                  frame_embeds.embeddings[vid].append(embeds[i0:it])
   
                if len(frame_embeds.embeddings.keys()) > 1: # n-1 videos fully embedded (100% certainty)
                  frame_embeds.save_vids(list(frame_embeds.embeddings.keys())[:-1])

            if ret and (ind % take_every_nth == 0):
                frame_embeds.frames[fname].append(preprocess(frame))
                # batch.append(preprocess(frame))
            ind += 1

        # video_embeddings = video_embeddings[:counter]
        # np.save(dst, video_embeddings)

    # Process remaining vids
    if frame_embeds._frame_count() > 0:
      batch, inds = frame_embeds.get_batch()
      batch = batch.to(device)
      embeds = model.encode_image(batch).cpu().detach().numpy()
      for vid, i0, it  in inds:
        frame_embeds.embeddings[vid].append(embeds[i0:it])
    frame_embeds.save_vids(list(frame_embeds.embeddings.keys())) # save remaining vids


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clip-video-encode.py video.mp4 embeddings.npy")
        sys.exit(1)

    clip_video_encode(sys.argv[1], sys.argv[2])
