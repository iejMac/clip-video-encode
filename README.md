# clip-video-encode
[![pypi](https://img.shields.io/pypi/v/clip-video-encode.svg)](https://pypi.python.org/pypi/clip-video-encode)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rom1504/clip-video-encode/blob/master/notebook/clip-video-encode.ipynb)
[![Try it on gitpod](https://img.shields.io/badge/try-on%20gitpod-brightgreen.svg)](https://gitpod.io/#https://github.com/rom1504/clip-video-encode)

Easily compute clip embeddings from video frames.

## Install

Using pip:
```
pip install clip-video-encode
```

Or build from source:
```
python setup.py install
```

## Usage 
```
NAME
    clip-video-encode - Encode frames using CLIP image encoder

SYNOPSIS
    clip-video-encode SRC <flags>

DESCRIPTION
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
      oc_model_name:
        str: open_clip model name, used for selecting CLIP architecture
      pretrained:
        str: open_clip pretrained weights name

POSITIONAL ARGUMENTS
    SRC

FLAGS
    --dest=DEST
        Default: ''
    --output_format=OUTPUT_FORMAT
        Default: 'files'
    --take_every_nth=TAKE_EVERY_NTH
        Default: 1
    --frame_workers=FRAME_WORKERS
        Default: 1
    --frame_memory_size=FRAME_MEMORY_SIZE
        Default: 4
    --metadata_columns=METADATA_COLUMNS
        Default: ''
    --use_dst_name=USE_DST_NAME
        Default: False
    --distribute=DISTRIBUTE
        Default: 'none'
    --oc_model_name=OC_MODEL_NAME
        Default: 'ViT-B-32'
    --pretrained=PRETRAINED
        Default: 'laion2b_s34b_b79k'
```

## API

This module exposes a single function `clip_video_encode` which takes the same arguments as the command line tool:
```python
import glob
from clip_video_encode import clip_video_encode

VIDS = glob.glob("some/path/my_videos/*.mp4")
EMBEDDING_DIR = "some/path/my_embeddings"
take_every_5 = 5

clip_video_encode(VIDS, EMBEDDING_DIR, take_every_5)
```

## Who is using clip-video-encode?
* [CLIP-Kinetics700](https://huggingface.co/datasets/iejMac/CLIP-Kinetics700) - The Kinetics700 dataset (700GB) can be compressed to ~8GB using clip-video-encode at 1 FPS
* [CLIP-WebVid](https://huggingface.co/datasets/iejMac/CLIP-WebVid) - The WebVid dataset (10M videos) encoded as CLIP ViT-B/32 embeddings at 1 FPS.

## Examples
Check out some cool clip-video-encode examples:
* [Thing detector](https://github.com/iejMac/clip-video-encode/tree/main/examples/thing_detector) - Look for things in videos using clip-video-encode generated embeddings.
* [Large dataset processing](https://github.com/iejMac/clip-video-encode/tree/main/clip_video_encode/dataset) - If you want to process a large dataset (like WebVid) into CLIP embeddings see the example at the bottom of the linked README.md.

Setup a virtualenv:

```
python3 -m venv .env
source .env/bin/activate
pip install -e .
```

to run tests:
```
pip install -r requirements-test.txt
```
then 
```
make lint
make test
```

You can use `make black` to reformat the code

`python -m pytest -x -s -v tests -k "dummy"` to run a specific test
