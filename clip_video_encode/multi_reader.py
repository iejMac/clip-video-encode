import os
import time

import cv2
import youtube_dl
import numpy as np

from torch.nn import Identity

from multiprocessing import shared_memory
from multiprocessing.pool import ThreadPool


QUALITY = "360p"
MAX_THREAD_COUNT = 12
POSTPROC_SHAPE = (224, 224, 3)

#TODO: research faster VideoReaders
# https://towardsdatascience.com/lightning-fast-video-reading-in-python-c1438771c4e6
# - VidGear https://pypi.org/project/vidgear/
# - torchvision.io.VideoReader

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class VideoReader:
  def __init__(
    self,
    take_every_nth=1,
  ):
    self.take_every_nth = take_every_nth

  def read_vids(self, vids, queue, comp_queue, chunk_size=1):

    shms = []

    while len(vids) > 0:

      tot_start_time = time.perf_counter()
      start_time = time.perf_counter()

      vid_chunk = vids[:chunk_size]
      vids = vids[chunk_size:]

      frams = {}

      with ThreadPool(MAX_THREAD_COUNT) as pool:

        def generate_frames(vid):

          # TODO : back to list
          video_frames = {
            "frames": [],
            "shape": None,
          }

          if not vid.endswith(".mp4"):  # youtube link
              ydl_opts = {}
              ydl = youtube_dl.YoutubeDL(ydl_opts)
              info = ydl.extract_info(vid, download=False)
              formats = info.get("formats", None)
              f = None
              for f in formats:
                  if f.get("format_note", None) != QUALITY:
                      continue
                  break

              cv2_vid = f.get("url", None)

              dst_name = info.get("id") + ".npy"
              # dst = dst_name if self.dest is None else os.path.join(self.dest, dst_name)

          else:
              cv2_vid = vid
              dst_name = vid[:-4].split("/")[-1] + ".npy"
              # dst = vid[:-4] + ".npy" if self.dest is None else os.path.join(self.dest, dst_name)

          cap = cv2.VideoCapture(cv2_vid)  # pylint: disable=I1101
          if not cap.isOpened():
              print(f"Error: {vid} not opened")
              # continue
              return video_frames, dst_name

          ret = True
          ind = 0
          while ret:
              ret, frame = cap.read()

              if ret and (ind % self.take_every_nth == 0):
                  # TODO: HACKY
                  if video_frames["shape"] is None:
                      cur_shape = list(frame.shape)[:2]
                      sm_ind, bg_ind = (0, 1) if cur_shape[0] < cur_shape[1] else (1, 0)
                      ratio = cur_shape[sm_ind] / 224
                      n_shape = cur_shape
                      n_shape[sm_ind] = 224
                      n_shape[bg_ind] = int(n_shape[bg_ind]/ratio)
                      video_frames["shape"] = tuple(n_shape)

                  # Resize:
                  frame = cv2.resize(frame, (video_frames["shape"][1], video_frames["shape"][0]), interpolation=cv2.INTER_CUBIC)
                  # Center crop:
                  my = int((video_frames["shape"][0] - 224)/2)
                  mx = int((video_frames["shape"][1] - 224)/2)

                  frame = frame[my:frame.shape[0]-my, mx:frame.shape[1]-mx]
                  frame = frame[:224, :224]

                  video_frames["frames"].append(frame)

              ind += 1

          frams[dst_name] = video_frames

        for _ in pool.imap_unordered(generate_frames, vid_chunk):
          pass

      fram_time = time.perf_counter() - start_time

      # FRAMS HAS WHAT WE NEED
      ind_dict = {}
      frame_count = 0
      max_h, max_w = 0, 0
      for k, v in frams.items():
        ind_dict[k] = (frame_count, frame_count + len(v["frames"]))
        frame_count += len(v["frames"])

      print(f"Frame FPS: {frame_count/fram_time}")

      full_shape = (frame_count, 224, 224, 3)

      mem_size = frame_count * full_shape[0] * full_shape[1] * full_shape[2] # using np.uint8 (images)
      shm = shared_memory.SharedMemory(create=True, size=mem_size)

      in_arr = np.ndarray(full_shape, dtype=np.uint8, buffer=shm.buf)
      for k, v in frams.items():
        i0, it = ind_dict[k]
        in_arr[i0:it] = v["frames"] # set video in top right of full_arr

      info = {
        "ind_dict": ind_dict,
        "shm_name": shm.name,
        "arr_shape": in_arr.shape,
      }

      print(f"Put {in_arr.shape} on queue")
      shms.append(shm)
      shm.close()

      fin_time = time.perf_counter() - tot_start_time
      print(f"FULL READ FPS: {frame_count/fin_time}")

      queue.put(info)

    queue.put("DONE_READING")

    # Wait for DONE_MAPPING
    msg = comp_queue.get()
    if msg != "DONE_MAPPING":
        print("Error: Message wrong message received")

    for shm in shms:
        shm.unlink()
