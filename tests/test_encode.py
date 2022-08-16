import os
import numpy as np
import pytest
import tempfile

from clip_video_encode import clip_video_encode

FRAME_COUNTS = {
    "vid1.mp4": 56,
    "vid2.mp4": 134,
    "https://www.youtube.com/watch?v=a8DM-tD9w2I": 20,
}


def test_encode():
    test_path = "tests/test_videos"
    with tempfile.TemporaryDirectory() as tmpdir:
        clip_video_encode(
            os.path.join(test_path, "test_list.parquet"),
            tmpdir,
            output_format="files",
            take_every_nth=2,
            frame_memory_size=0.125,
            metadata_columns=["caption", "meta"],
            use_dst_name=True,
        )
        assert len(os.listdir(tmpdir)) == len(FRAME_COUNTS) * 3
        for vid in FRAME_COUNTS.keys():
            if vid.endswith(".mp4"):
                ld = vid[:-4] + ".npy"
            else:
                ld = vid.split("=")[-1] + ".npy"

            embeddings = np.load(os.path.join(tmpdir, ld))
            assert embeddings.shape[0] == FRAME_COUNTS[vid] // 2  # frame count
            assert embeddings.shape[1] == 512  # embed dim
