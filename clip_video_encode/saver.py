import cv2
import numpy as np

from torch.nn import Identity


def save_videos(vids, frames, dest=None, preprocess=Identity(), take_every_nth=1):
  '''
    vids: list of videos to encode
    frames: list where frames are stored
  '''

  for vid in vids:
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

          vid = f.get("url", None)

          dst_name = info.get("id") + ".npy"
          dst = dst_name if dest is None else os.path.join(dest, dst_name)
      else:
          dst_name = vid[:-4].split("/")[-1] + ".npy"
          dst = vid[:-4] + ".npy" if dest is None else os.path.join(dest, dst_name)

      cap = cv2.VideoCapture(vid)  # pylint: disable=I1101
      if not cap.isOpened():
          print("Error: Video not opened")
          sys.exit(1)

      ret = True 
      ind = 0
      while ret:
          ret, frame = cap.read()

          if ret and (ind % take_every_nth == 0):
              frames.append((dst, preprocess(frame))) # (dest, frame_from_vid)
          ind += 1
