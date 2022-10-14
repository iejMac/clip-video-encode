import os
import numpy as np
import pytest
import tempfile
import torch
import open_clip

from clip_video_encode import clip_video_encode


def test_similarity(video):
    test_path = "tests/test_videos"
    with tempfile.TemporaryDirectory() as tmpdir:
        clip_video_encode(
            ["tests/test_videos/vid1.mp4", "tests/test_videos/vid2.mp4"],
            tmpdir,
            output_format="files",
            take_every_nth=2,
            frame_memory_size=0.125,
            metadata_columns=["caption", "meta"],
            use_dst_name=True,
        )

        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        text = open_clip.tokenize(["bears", "monkey"])
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_feat = model.encode_text(text)

            for vid in ["vid1.mp4", "vid2.mp4"]
                frame_embeddings = np.load(os.path.join(tmpdir, vid[:-4] + ".npy"))
                frame_feat = torch.from_numpy(frame_embeddings[0]) # only take first frame

                text_probs = (100.0 * frame_feat @ text_feat.T).softmax(dim=-1)

                best = torch.argmax(text_probs)
                assert best == (0 if vid == "vid1.mp4" else 1)
