import os
import numpy as np
import pytest
import tempfile

from clip_video_encode import clip_video_encode

FRAME_COUNTS = {
    "vid1.mp4": 74,
    "vid2.mp4": 179,
}


def test_encode():
    test_path = "tests/test_videos"
    vids = os.listdir(test_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        for vid in vids:
            src = os.path.join(test_path, vid)
            dst = os.path.join(tmpdir, vid[:-4] + ".npy")
            clip_video_encode(src, dst)

            embeddings = np.load(dst)
            assert embeddings.shape[0] == FRAME_COUNTS[vid]  # frame count
            assert embeddings.shape[1] == 512  # embed dim
