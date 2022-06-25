# video-embedding-dataset

In this document we define a dataset format for training contrastive language-video temporal aggregator models. This is a very early-stage project so please feel free to propose ideas/changes.

## Overview

The data processing pipeline for video-embedding-datasets is: raw dataset -> organized dataset (some common format so clip-video-encode handle it nicely) -> webdataset


## Raw dataset

You can find a list of raw datasets in:
https://docs.google.com/document/d/12zYnjZabR2e17vPO2XpctIf1qUQeEX7kYC8GdDqWM-k/edit

## Processed datasets

We us the [webdataset](https://github.com/webdataset/webdataset) format therefore the final processed dataset should be in the form of a list of tar files split up into train/val/test splits. Each tar file should have:
* numpy array (.npy) of shape (frame_count, embed_dim) where embed_dim is 512 for now
* text caption (.cap) that describes the video
* json file with any additional metadata (YouTube video ID, time window of video clip from larger video, etc.)


```
......
	video-embedding-dataset
   ├── train
	 │   ├── ds_00000.tar
	 |   |     ├── name0.npy
	 |   |     ├── name0.cap
	 |   |     ├── name0.json
	 |   |     ├── name1.npy
	 |   |     ├── name1.cap
	 |   |     ├── name1.json
	 |   |     └── ...
	 │   ├── ds_00001.tar
	 |   |     ├── name0.npy
	 |   |     ├── name0.cap
	 |   |     ├── name0.json
	 |   |     └── ...
	 │   └── ...
	 │
   ├── val
	 │   ...
   ├── test
	 │   ...
```

## Example of preparing dataset:

Make this easier with clip-video-encode i.e.:
1. Raw DS
2. split up videos into train/val/test
3. clip-video-encode does the rest

## Example of prepared datasets:
Examples: 
* https://huggingface.co/datasets/iejMac/CLIP-Kinetics700
