import os
import glob
import pytest
import tempfile

import clip
import multiprocessing
import numpy as np
import tarfile
import torch

from torchvision.transforms import Compose, Normalize, ToPILImage, ToTensor

from clip_video_encode.utils import block2dl
from clip_video_encode.simplemapper import FrameMapper
from clip_video_encode.writer import FileWriter, WebDatasetWriter


FRAME_COUNTS = {
    "vid1.mp4": 56,
    "vid2.mp4": 134,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def test_utils():
    n_px = 224
    prepro = Compose(
        [
            ToPILImage(),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    post_prepro_shape = (3, n_px, n_px)

    N_FRAMES = 100
    block = np.zeros((100, n_px, n_px, 3), dtype=np.uint8)

    BATCH_SIZE = 20
    N_DATASET_WORKERS = multiprocessing.cpu_count()

    fr_dl = block2dl(block, prepro, BATCH_SIZE, N_DATASET_WORKERS)

    batch_count = 0
    for batch in fr_dl:
        assert batch.shape == (BATCH_SIZE, *post_prepro_shape)
        batch_count += 1
    assert batch_count == int(N_FRAMES / BATCH_SIZE)


def test_mapper():
    # Initialize model:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    model_input_shape = (3, 224, 224)
    model_output_dim = 512

    fm = FrameMapper(model, device)

    bs = 20
    batch = torch.rand(bs, *model_input_shape).to(device)
    output = fm(batch)
    assert output.shape == (bs, model_output_dim)


@pytest.mark.parameterize("writer_type", ["files", "webdataset"])
def test_writer(writer_type):
    with tempfile.TemporaryDirectory() as tmpdir:
        if writer_type == "files":
            writer = FileWriter(tmpdir)
        elif writer_type == "webdataset":
            writer = WebDatasetWriter(tmpdir, 0, 5, "npy")

        N_VIDS = 5
        N_FRAMES = 100
        lat_dim = 8

        vid_embeds = [np.ones((N_FRAMES, lat_dim), dtype=float) * i for i in range(N_VIDS)]

        for i, emb in enumerate(vid_embeds):
            writer.write(emb, f"{i}.npy")

        if writer_type == "files":
            for i in range(N_VIDS):
                dst_name = f"{i}.npy"
                np_embeddings = np.load(os.path.join(tmpdir, dst_name))
                assert np_embeddings.shape == (N_FRAMES, lat_dim)

                val = int(dst_name[0])
                assert np.all(np_embeddings == val)
        elif writer_type == "webdataset":
            l = glob.glob(tmpdir + "/*.tar")
            assert len(l) == 1
            assert l[0] == tmpdir + "/00000.tar":
            assert len(tarfile.open(tmpdir + "/00000.tar").getnames()) == N_VIDS
