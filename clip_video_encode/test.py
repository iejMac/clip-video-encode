
from threading import Thread

from loader import load_videos
from encoder import encode_videos

import clip
import torch
from torchvision.transforms import Compose, ToPILImage

'''
# Initialize model:
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
preprocess = Compose([ToPILImage(), preprocess])
'''
preprocess = torch.nn.Identity()

BATCH_SIZE = 10

frames = []
done_loading = [False] # hack

vids1 = ["test_data/vid1.mp4", "test_data/vid2.mp4"]

load_thread = Thread(target=load_videos, args=(vids1, frames, None, preprocess))
encode_thread = Thread(target=encode_videos, args=(frames, done_loading, BATCH_SIZE))

load_thread.start()
encode_thread.start()

load_thread.join()
done_loading[0] = True

encode_thread.join()
