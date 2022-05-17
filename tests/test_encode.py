import os
import numpy as np
import pytest
import tempfile

from clip_video_encode import clip_video_encode

FRAME_COUNTS = {
    "vid1.mp4": 56,
    "vid2.mp4": 134,
    "https://www.youtube.com/watch?v=EKtBQbK4IX0": 78,
}


def test_encode():
    test_path = "tests/test_videos"
    with tempfile.TemporaryDirectory() as tmpdir:
        for vid in FRAME_COUNTS.keys():
            if vid.endswith(".mp4"):
                src = os.path.join(test_path, vid)
                dst = os.path.join(tmpdir, vid[:-4] + ".npy")
            else:
                src = vid
                dst = os.path.join(tmpdir, vid.split("=")[-1] + ".npy")

            clip_video_encode(src, dst)

            embeddings = np.load(dst)
            assert embeddings.shape[0] == FRAME_COUNTS[vid]  # frame count
            assert embeddings.shape[1] == 512  # embed dim
