# clip-video-encode
[![pypi](https://img.shields.io/pypi/v/clip-video-encode.svg)](https://pypi.python.org/pypi/clip-video-encode)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rom1504/clip-video-encode/blob/master/notebook/clip-video-encode.ipynb)
[![Try it on gitpod](https://img.shields.io/badge/try-on%20gitpod-brightgreen.svg)](https://gitpod.io/#https://github.com/rom1504/clip-video-encode)

Easily compute clip embeddings from video frames (mp4 or youtube links)

## Install
```
pip install clip-video-encode
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
      take_every_nth:
        int: only take every nth frame

    Output:
      None

POSITIONAL ARGUMENTS
    SRC

FLAGS
    --dest=DEST
        Type: Optional[]
        Default: None
    --take_every_nth=TAKE_EVERY_NTH
        Default: 1
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

## For development

Either locally, or in [gitpod](https://gitpod.io/#https://github.com/rom1504/clip-video-encode) (do `export PIP_USER=false` there)

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
