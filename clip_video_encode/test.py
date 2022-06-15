import numpy as np
import torch
import time
import glob

import clip

from multiprocessing import SimpleQueue, Process, set_start_method, shared_memory
from torchvision.transforms import ToPILImage, Compose, ToTensor, Normalize

# from reader_ffmpeg import read_vids
from reader import read_vids

from simplemapper import FrameMapper
from writer import write_embeddings

from batcher import get_dl

def _convert_image_to_rgb(image):
    return image.convert("RGB")

VID_DIR = "test_data/*.mp4"

EMB_DIR = "test_npy"

if __name__ == "__main__":

  vids = glob.glob(VID_DIR)
  vids = vids[:1]

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, _ = clip.load("ViT-B/32", device=device)
  preproc = Compose([ToPILImage(), _convert_image_to_rgb, ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

  info_q = SimpleQueue()
  complete_q = SimpleQueue() # TODO: SharedMemory hack, do properly

  fm = FrameMapper(model)

  N_VIDS = 100
  BATCH_SIZE = 256
  N_DATASET_WORKERS = 8
  take_every_nth = 25

  vr_proc = Process(target=read_vids, args=(vids, info_q, complete_q, N_VIDS, take_every_nth))

  tot_start_time = time.perf_counter()
  vr_proc.start()

  TOT_FRAME_COUNT = 0
  while True:
    info = info_q.get()

    if isinstance(info, str):
        break

    shm = shared_memory.SharedMemory(name=info["shm_name"])
    block = np.ndarray((info["frame_count"], 224, 224, 3), dtype=np.uint8, buffer=shm.buf)

    dl = get_dl(block, preproc, BATCH_SIZE, N_DATASET_WORKERS)

    start_time = time.perf_counter()

    embeddings = []
    for batch in dl:
      with torch.no_grad():
        emb = fm(batch.to(device))
        embeddings.append(emb)

    embeddings = np.concatenate(embeddings)
    frame_count = len(embeddings)

    proc_time = time.perf_counter() - start_time
    print(f"PROC FPS: {frame_count/proc_time}")


    start_time = time.perf_counter()
    write_embeddings(info["ind_dict"], embeddings, EMB_DIR)
    write_time = time.perf_counter() - start_time
    print(f"WRITE FPS: {frame_count/write_time}")

    TOT_FRAME_COUNT += frame_count
    shm.close()

  complete_q.put("DONE_MAPPING") # TODO: SharedMemory hack, do properly

  vr_proc.join()

  read_time = time.perf_counter() - tot_start_time
  print(read_time)
  print(f"FULL PROCESS FPS: {TOT_FRAME_COUNT/read_time}")
