import numpy as np
import torch
import time
import glob

import clip

from multiprocessing import SimpleQueue, Process, set_start_method, shared_memory
from torchvision.transforms import ToPILImage, Compose, ToTensor, Normalize

from multi_reader import VideoReader
from simplemapper import FrameMapper

from batcher import HelperDataset, ds_to_dl

def _convert_image_to_rgb(image):
    return image.convert("RGB")

VID_DIR = "/home/iejmac/test_vids/*.mp4"


if __name__ == "__main__":

  vids = glob.glob(VID_DIR)
  vids = vids[:200]

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, _ = clip.load("ViT-B/32", device=device)
  preproc = Compose([ToPILImage(), _convert_image_to_rgb, ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

  info_q = SimpleQueue()
  complete_q = SimpleQueue() # TODO: SharedMemory hack, do properly

  vr = VideoReader(take_every_nth=1)
  fm = FrameMapper(model)

  N_VIDS = 50
  vr_proc = Process(target=vr.read_vids, args=(vids, info_q, complete_q, N_VIDS))

  tot_start_time = time.perf_counter()
  vr_proc.start()

  TOT_FRAME_COUNT = 0
  while True:
    info = info_q.get()

    if isinstance(info, str):
        break

    shm = shared_memory.SharedMemory(name=info["shm_name"])
    block = np.ndarray(info["arr_shape"], dtype=np.uint8, buffer=shm.buf) 

    ds = HelperDataset(block, preproc)
    dl = ds_to_dl(ds, 256, 8)

    frame_count = 0
    start_time = time.perf_counter()
    for batch in dl:
      with torch.no_grad():
        emb = fm(batch.to("cuda"))
      frame_count += batch.shape[0]

    proc_time = time.perf_counter() - start_time
    print(f"PROC FPS: {frame_count/proc_time}")
    TOT_FRAME_COUNT += frame_count

    shm.close()

  complete_q.put("DONE_MAPPING") # TODO: SharedMemory hack, do properly
  vr_proc.join()

  read_time = time.perf_counter() - tot_start_time
  print(read_time)
  print(f"FULL PROCESS FPS: {TOT_FRAME_COUNT/read_time}")
