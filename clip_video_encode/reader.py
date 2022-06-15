import os
import time

import cv2
import youtube_dl
import numpy as np

from torch.nn import Identity

from multiprocessing import shared_memory
from multiprocessing.pool import ThreadPool


QUALITY = "360p"
THREAD_COUNT = 12
POSTPROC_SHAPE = (224, 224, 3)
IMG_SIDE = 224


def read_vids(vids, queue, termination_queue=None, chunk_size=1, take_every_nth=1):

    shms = []

    while len(vids) > 0:

      tot_start_time = time.perf_counter()
      start_time = time.perf_counter()

      vid_chunk = vids[:chunk_size]
      vids = vids[chunk_size:]

      frams = {}

      with ThreadPool(THREAD_COUNT) as pool:

        def generate_frames(vid):

          video_frames = []

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
              return

          ret = True
          frame_shape = None
          ind = 0
          while ret:
              ret, frame = cap.read()

              if ret and (ind % take_every_nth == 0):
                  # NOTE: HACKY
                  if frame_shape is None:
                      cur_shape = list(frame.shape)[:2]
                      sm_ind, bg_ind = (0, 1) if cur_shape[0] < cur_shape[1] else (1, 0)
                      ratio = cur_shape[sm_ind] / IMG_SIDE
                      n_shape = cur_shape
                      n_shape[sm_ind] = IMG_SIDE
                      n_shape[bg_ind] = int(n_shape[bg_ind]/ratio)
                      frame_shape = tuple(n_shape)

                  # Resize:
                  frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_CUBIC)
                  # Center crop:
                  my = int((frame_shape[0] - IMG_SIDE)/2)
                  mx = int((frame_shape[1] - IMG_SIDE)/2)

                  frame = frame[my:frame.shape[0]-my, mx:frame.shape[1]-mx]
                  frame = frame[:IMG_SIDE, :IMG_SIDE]

                  video_frames.append(frame)

              ind += 1

          frams[dst_name] = video_frames

        for _ in pool.imap_unordered(generate_frames, vid_chunk):
          pass

      fram_time = time.perf_counter() - start_time

      ind_dict = {}
      frame_count = 0
      max_h, max_w = 0, 0
      for k, v in frams.items():
        ind_dict[k] = (frame_count, frame_count + len(v))
        frame_count += len(v)

      print(frame_count, fram_time)
      print(f"Frame FPS: {frame_count/fram_time}")

      full_shape = (frame_count, *POSTPROC_SHAPE)

      mem_size = frame_count * full_shape[0] * full_shape[1] * full_shape[2]
      shm = shared_memory.SharedMemory(create=True, size=mem_size)

      in_arr = np.ndarray(full_shape, dtype=np.uint8, buffer=shm.buf)
      for k, v in frams.items():
        i0, it = ind_dict[k]
        in_arr[i0:it] = v

      info = {
        "ind_dict": ind_dict,
        "shm_name": shm.name,
        "frame_count": frame_count,
      }

      print(f"Put {in_arr.shape} on queue")
      shms.append(shm)
      shm.close()

      fin_time = time.perf_counter() - tot_start_time
      print(f"FULL READ FPS: {frame_count/fin_time}")

      queue.put(info)

    queue.put("DONE_READING")

    # Wait for DONE_MAPPING
    if termination_queue is not None:
      msg = termination_queue.get()
      if msg != "DONE_MAPPING":
          print("Error: Message wrong message received")

    for shm in shms:
        shm.unlink()
