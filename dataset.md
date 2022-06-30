# video-embedding-dataset

In this document we define a dataset format for training contrastive language-video temporal aggregator models. This is a very early-stage project so please feel free to propose ideas/changes.

## Overview

The data processing pipeline for video-embedding-datasets is: raw dataset -> processed dataset (some common format so clip-video-encode can handle it nicely) -> webdataset


## Raw dataset

You can find a list of raw datasets in:
https://docs.google.com/document/d/12zYnjZabR2e17vPO2XpctIf1qUQeEX7kYC8GdDqWM-k/edit

## Processed dataset

A processed dataset is the common format we use before combining all the files into a Embedding WebDataset using our [create_shards.py]() script. This format is a directory with 3 subdirectories for each of the train/val/test splits. Inside each subdirectory there should be n triples with unique names:

* numpy files with the embeddings of the frames (generated using clip-video-encode) of shape (frame_count, embed_dim)
* txt files with the caption for that video
* json files with the metadata for that video

```
processed_dataset
 ├── train
 |   ├── name0.npy
 |   ├── name0.txt
 |   ├── name0.json
 |   ├── name1.npy
 |   ├── name1.txt
 |   ├── name1.json
 |   └── ...
 ├── val
 |   ├── name2.npy
 |   ├── name2.txt
 |   ├── name2.json
 │   ...
 ├── test
 │   ...
```

## Embedding WebDataset 

We us the [webdataset](https://github.com/webdataset/webdataset) format therefore the final processed dataset should be in the form of a list of tar files with unique ID's and a splits.csv file describing which tar files belong to which splits. Each tar file should have 10000 triples (final tar in each split might have <10000) with the same name and ID:

* numpy array (.npy) of shape (frame_count, embed_dim) where embed_dim is 512 for now
* text caption (.txt) that describes the video
* json file (.json) with any additional metadata (YouTube video ID, time window of video clip from larger video, etc.)

Note: zero-padding of names may vary from dataset to dataset.
```
video-embedding-dataset
 ├── splits.csv
 ├── ds_00000.tar
 |     ├── vid_00000.npy
 |     ├── vid_00000.txt
 |     ├── vid_00000.json
 |     ├── vid_00001.npy
 |     ├── vid_00001.txt
 |     ├── vid_00001.json
 |     └── ...
 ├── ds_00001.tar
 |     ├── vid_00002.npy
 |     ├── vid_00002.txt
 |     ├── vid_00002.json
 │     ...
 ...
```

## Example of preparing dataset:

Make this easier with clip-video-encode i.e.:
1. Raw DS
2. split up videos into train/val/test
3. clip-video-encode does the rest

## Example of prepared Embedding WebDatasets:
Examples: 
* https://huggingface.co/datasets/iejMac/CLIP-Kinetics700
