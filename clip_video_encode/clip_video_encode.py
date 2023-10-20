"""encode video with CLIP"""
import sys
import time

import math
import numpy as np
import torch

from video2numpy.frame_reader import FrameReader

from .reader import Reader, read_shard
from .simplemapper import FrameMapper
from .utils import block2dl
from .writer import FileWriter, WebDatasetWriter
from .distributed import world_info_from_env
from .handle_chunk import encode_chunk

import tarfile
import tempfile
import os
import json
import braceexpand
import glob
import fsspec
import io

CHUNK_SIZE = 100


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def clip_video_encode(
    src,
    dest="",
    output_format="files",
    take_every_nth=25,
    target_fps=-1,
    input_format="table",
    frame_workers=1,
    frame_memory_size=4,
    metadata_columns="",
    use_dst_name=False,
    distribute="none",
    oom_shard_count=5,
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    captioning_strategy="none",
    frame_tokenization_strategy="none",
    generated_caption_key="generated_caption",  # this will put it in json, make this 'caption' if you want it in txt
    pass_through_keys="mp4,txt,json",
    caption_similarity=False,
    img_size=224,
):
    """
    Encode frames using CLIP image encoder

    Input:
      src:
        str: path to mp4 file
        str: youtube link
        str: path to txt file with multiple mp4's or youtube links
        list: list with multiple mp4's or youtube links
      dest:
        str: directory where to save embeddings to
        None: dest = src + .npy
      output_format:
        str: "files" or "webdataset"
      take_every_nth:
        int: only take every nth frame
      target_fps:
        int: target fps to downsample videos to (-1 means original fps or take_every_nth)
      frame_workers:
        int: number of Processes to distribute video reading to.
      frame_memory_size:
        int: GB of memory for FrameReader.
      metadata_columns:
        str: a comma separated list of metadata column names to look for in src
      use_dst_name:
        bool: use the save name suggested by video2numpy
      distribute:
        str: distribution strategy, currently either slurm or none
      model_name:
        str:
          - open_clip model name, used for selecting CLIP architecture
          - vqgan config path
      pretrained:
        str:
          - open_clip pretrained weights name
          - vqgan weights checkpoint path
      captioning_strategy:
        str: which frames of a video to generate captions for. Possible values are:
          - none: don't generate any captions
          - center: generate a caption for the middle frame
        int: (NOT IMPLEMENTED) step size for which frames to generate captions for
      pass_through_keys:
        str: comma separated list of extension to pass through from input dataset (if webdataset format)
      caption_similarity:
        bool: whether to put the similarity between the average frame embedding and text embedding into metadata
      img_size:
        int: pixel height and width of target output shape
    """
    assert input_format in ["table", "webdataset"]

    if isinstance(metadata_columns, str):
        metadata_columns = [metadata_columns] if metadata_columns != "" else []
    metadata_columns = list(metadata_columns) if isinstance(metadata_columns, tuple) else metadata_columns

    if isinstance(pass_through_keys, str):
        pass_through_keys = pass_through_keys.split(",")

    if input_format == "table":
        reader = Reader(src, metadata_columns)
        vids, ids, meta = reader.get_data()
        meta_refs = list(range(len(vids)))

    else:  # WebDataset, so we distribute shards
        shards = list(braceexpand.braceexpand(src))

        # NOTE: this might need to be improved, some shards may not be complete
        fs, output_path = fsspec.core.url_to_fs(dest)
        if not fs.exists(output_path):
            fs.mkdir(output_path)
            done_shards = set()
        else:
            done_shards = set(int(x.split("/")[-1].split("_")[0]) for x in fs.glob(output_path + "/*.tar"))

        print(f"Removing {len(done_shards)} done_shards from processing queue...")
        s_ids = [s.split("/")[-1][: -len(".tar")] for s in shards]
        shards = [s for s_id, s in zip(s_ids, shards) if int(s_id) not in done_shards]

    starting_shard_id = 0
    shard_sample_count = 10000

    if distribute == "slurm":
        local_rank, global_rank, world_size = world_info_from_env()
        if input_format == "table":
            work_size = math.ceil(len(vids) / world_size)
        else:
            work_size = math.ceil(len(shards) / world_size)
        print(f"Slurm worker {global_rank} processing {work_size} videos...")
        ws, wf = global_rank * work_size, (global_rank + 1) * work_size
        if input_format == "table":
            vids = vids[ws:wf]
            ids = ids[ws:wf]
            for mc in meta.keys():
                meta[mc] = meta[mc][ws:wf]

            starting_shard_id += math.ceil(work_size / shard_sample_count) * global_rank
        elif input_format == "webdataset":
            shards = shards[ws:wf]
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        local_rank, global_rank, world_size = 0, 0, 1  # TODO: how do we do this?
        device = "cuda" if torch.cuda.is_available() else "cpu"

    assert output_format in ["files", "webdataset"]
    if output_format == "files":
        writer = FileWriter(dest)
    elif output_format == "webdataset":
        # TODO: maybe include params for this?
        starting_shard_id = int(shards[0].split("/")[-1].split(".tar")[0])
        writer = WebDatasetWriter(dest, oom_shard_count, "npy", maxcount=1e6, shard_id=starting_shard_id)

    fm = FrameMapper(
        model_name,
        pretrained,
        device,
        get_text_tokenizer=(caption_similarity or (captioning_strategy != "none")),
        get_frame_tokenizer=(frame_tokenization_strategy != "none"),
    )

    if input_format == "table":
        fr = FrameReader(
            vids,
            meta_refs,
            take_every_nth=take_every_nth,
            target_fps=target_fps,
            resize_size=img_size,
            workers=frame_workers,
            memory_size=frame_memory_size,
        )
        fr.start_reading()

        frames, ind_dict = [], {}
        block_size = 0
        i = 0
        for vid_frames, info in fr:
            i += 1
            frames.append(vid_frames)
            ind_dict[info["reference"]] = (
                block_size,
                block_size + vid_frames.shape[0],
                info["dst_name"],
            )
            block_size += vid_frames.shape[0]

            if i % CHUNK_SIZE == 0:
                encode_chunk(frames, ind_dict, writer, fm, meta, ids, use_dst_name, device, input_format=input_format)
                frames, ind_dict, block_size = [], {}, 0

        if len(frames) > 0:  # TODO: make this cleaner
            encode_chunk(frames, ind_dict, writer, fm, meta, ids, use_dst_name, device, input_format=input_format)
    else:  # WebDataset shard logic
        for shard in shards:
            # try:
            times = {}
            t = time.time()
            with tempfile.TemporaryDirectory(prefix=f"worker_{global_rank}_") as tempdir:
                os.chmod(tempdir, 0o777)  # This lets subprocesses from v2np read files in the tempdir
                folder = "/".join(shard.split("/")[0:-1])
                fs, output_path = fsspec.core.url_to_fs(folder)

                shard_id = shard.split("/")[-1]
                tar_bytes = io.BytesIO(fs.open(f"{output_path}/{shard_id}").read())
                with tarfile.open(fileobj=tar_bytes) as tar:
                    tar.extractall(tempdir)
                writer.create_shard(shard_id=int(shard_id.split(".tar")[0]))
                times["download_and_extract"] = times.get("download_and_extract", 0) + time.time() - t
                t = time.time()
                vids, ids, meta = read_shard(tempdir, pass_through_keys=pass_through_keys)
                meta_refs = list(range(len(vids)))
                fr = FrameReader(
                    vids,
                    meta_refs,
                    take_every_nth=take_every_nth,
                    target_fps=target_fps,
                    resize_size=img_size,
                    workers=frame_workers,
                    memory_size=frame_memory_size,
                )
                fr.start_reading()

                frames, ind_dict = [], {}
                block_size = 0
                i = 0
                n_frames = 0
                for vid_frames, info in fr:
                    i += 1

                    if captioning_strategy == "center":
                        vid_frames = vid_frames[len(vid_frames) // 2 : len(vid_frames) // 2 + 1]

                    n_frames += len(vid_frames)
                    frames.append(vid_frames)
                    ind_dict[info["reference"]] = (
                        block_size,
                        block_size + vid_frames.shape[0],
                        info["dst_name"],
                    )
                    block_size += vid_frames.shape[0]
                    times["read_frames"] = times.get("read_frames", 0) + time.time() - t
                    t = time.time()

                    if i % CHUNK_SIZE == 0:
                        encode_chunk(
                            frames,
                            ind_dict,
                            writer,
                            fm,
                            meta,
                            ids,
                            use_dst_name,
                            device,
                            input_format=input_format,
                            captioning_strategy=captioning_strategy,
                            frame_tokenization_strategy=frame_tokenization_strategy,
                            generated_caption_key=generated_caption_key,
                        )
                        times["encode"] = times.get("encode", 0) + time.time() - t
                        t = time.time()
                        frames, ind_dict, block_size = [], {}, 0
                t = time.time()
                if len(frames) > 0:  # TODO: make this cleaner
                    encode_chunk(
                        frames,
                        ind_dict,
                        writer,
                        fm,
                        meta,
                        ids,
                        use_dst_name,
                        device,
                        input_format=input_format,
                        captioning_strategy=captioning_strategy,
                        frame_tokenization_strategy=frame_tokenization_strategy,
                        generated_caption_key=generated_caption_key,
                    )
                times["encode"] = times.get("encode", 0) + time.time() - t
                t = time.time()
            frame_adjusted = {k: n_frames / v for k, v in times.items()}
            print(f"Frames/s: {frame_adjusted}")
        # except Exception as e:  # pylint: disable=(broad-except)
        #     print(f"Shard {shard} failed: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python clip-video-encode.py video.mp4 embeddings.npy take_every_nth")
        sys.exit(1)
    clip_video_encode(sys.argv[1], sys.argv[2], int(sys.argv[3]))
