import os
import glob
import pytest
import tempfile

import clip
import numpy as np
import torch

from clip_video_encode import VideoReader, FrameBatcher, FrameMapper, EmbeddingWriter

from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToPILImage, ToTensor
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


FRAME_COUNTS = {
    "vid1.mp4": 56,
    "vid2.mp4": 134,
}


def test_reader():
    vids = glob.glob("tests/test_videos/*.mp4")

    vr = VideoReader()
    frames = vr.read_vids(vids)

    for vid, frs in frames.items():
        assert len(frs) == FRAME_COUNTS[vid[:-4] + ".mp4"]


def test_batcher():
  n_px = 224
  def _convert_image_to_rgb(image):
    return image.convert("RGB")
  prepro = Compose([
    ToPILImage(),
    Resize(n_px, interpolation=BICUBIC),
    CenterCrop(n_px),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
  ])
  post_prepro_shape = (3, n_px, n_px)

  fb = FrameBatcher(prepro)

  # Add fake videos:
  N_VIDS = 5
  N_FRAMES = 100
  for i in range(N_VIDS):
    fb.add_frames([np.zeros((640, 480, 3)).astype(np.uint8) + i for j in range(N_FRAMES)], str(i))

  BATCH_SIZE = 20
  fr_dl, vi = fb.get_dataloader(BATCH_SIZE, 0)

  for batch in fr_dl:
    assert batch.shape == (BATCH_SIZE, *post_prepro_shape)
  for vid, i0, it in vi:
    assert it - i0 == N_FRAMES


def test_mapper():
  # Initialize model:
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, _ = clip.load("ViT-B/32", device=device)

  model_input_shape = (3, 224, 224)
  model_output_dim = 512

  fm = FrameMapper(model)

  bs = 20
  batch = torch.rand(bs, *model_input_shape).to(device)
  output = fm(batch)
  assert output.shape == (bs, model_output_dim)


def test_writer():
  with tempfile.TemporaryDirectory() as tmpdir:
    ew = EmbeddingWriter(tmpdir)
    N_VIDS = 5
    N_FRAMES = 100

    lat_dim = 64

    for vid in range(N_VIDS):
      dst_name = str(vid) + ".npy"
      ew.init_video(dst_name, N_FRAMES)

      n_embeds = N_FRAMES - 1 if vid == N_VIDS-1 else N_FRAMES # last video incomplete
      ew.add_embeddings(dst_name, np.random.random((n_embeds, lat_dim)))
    ew.flush()

    assert len(np.concatenate(ew.embeddings["4.npy"])) == N_FRAMES - 1
    for vid in range(N_VIDS - 1):
      embeds = np.load(os.path.join(tmpdir, str(vid) + ".npy"))
      assert embeds.shape == (N_FRAMES, lat_dim)
