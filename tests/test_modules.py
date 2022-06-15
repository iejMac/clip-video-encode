import os
import glob
import pytest
import tempfile

import clip
import multiprocessing
import numpy as np
import torch

from torchvision.transforms import Compose, Normalize, ToPILImage, ToTensor

from clip_video_encode.batcher import get_dl
from clip_video_encode.reader import read_vids
from clip_video_encode.simplemapper import FrameMapper
from clip_video_encode.writer import write_embeddings


FRAME_COUNTS = {
    "vid1.mp4": 56,
    "vid2.mp4": 134,
}


def test_reader():
    vids = glob.glob("tests/test_videos/*.mp4")

    q = multiprocessing.SimpleQueue()

    read_vids(vids, q)

    while not q.empty():
        info = q.get()

        if isinstance(info, str):
            assert info == "DONE_READING"
        else:
            assert isinstance(info, dict)
            ind_dict = info["ind_dict"]

            frame_count = 0
            for vid, inds in ind_dict.items():
                mp4_name = vid[:-4] + ".mp4"

                i0, it = inds
                vid_frame_count = FRAME_COUNTS[mp4_name]
                assert it - i0 == vid_frame_count

                frame_count += vid_frame_count

            assert frame_count == info["frame_count"]


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def test_batcher():
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

    fr_dl = get_dl(block, prepro, BATCH_SIZE, N_DATASET_WORKERS)

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

    fm = FrameMapper(model)

    bs = 20
    batch = torch.rand(bs, *model_input_shape).to(device)
    output = fm(batch)
    assert output.shape == (bs, model_output_dim)


def test_writer():
    with tempfile.TemporaryDirectory() as tmpdir:
        N_VIDS = 5
        N_FRAMES = 100
        lat_dim = 8

        ind_dict = dict([(f"{i}.npy", (i * N_FRAMES, (i + 1) * N_FRAMES)) for i in range(N_VIDS)])
        vid_embeds = [np.ones((N_FRAMES, lat_dim), dtype=float) * i for i in range(N_VIDS)]
        embeddings = np.concatenate(vid_embeds)

        write_embeddings(ind_dict, embeddings, tmpdir)

        for dst_name, vid_inds in ind_dict.items():
            np_embeddings = np.load(os.path.join(tmpdir, dst_name))

            i0, it = vid_inds
            assert len(np_embeddings) == it - i0

            val = int(dst_name[0])
            assert np.all(np_embeddings == val)
