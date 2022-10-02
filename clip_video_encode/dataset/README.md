# video-embedding-dataset

In this document we define a dataset format for training contrastive language-video temporal embedding aggregator models.

## Overview

The data processing pipeline for video-embedding-datasets is: raw dataset -> organized csv/parquet table -> webdataset

## Raw dataset

You can find a list of raw datasets in:
https://docs.google.com/document/d/12zYnjZabR2e17vPO2XpctIf1qUQeEX7kYC8GdDqWM-k/edit

## Organized Tabular Format 

For clip-video-encode to create the EmbeddingWebDataset you need to provide it a format that contains all relevant videos and metadata. The preferred format is parquet. The only necessary columns are videoID - the unique ID for each video, and videoLoc - the location of the video (youtube link, mp4 link, or even path to mp4 on disc). Additionally, if you want to pair videos with text we include the option for a special "caption" column which will appropriately distinguish the associated text caption in the dataset. An example table is shown in [example_tab.csv](https://github.com/iejMac/clip-video-encode/tree/main/clip_video_encode/dataset/example_tab.csv").

## Embedding WebDataset 

We us the [webdataset](https://github.com/webdataset/webdataset) format therefore the final processed dataset should be in the form of a list of tar files with unique ID's and a splits.csv file describing which tar files belong to which splits (or alternatively just manually separate splits into directories). Each tar file should have 10000 triples (final tar in each split might have <10000):

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
 |     ├── vid_10000.npy
 |     ├── vid_10000.txt
 |     ├── vid_10000.json
 ├── ds_00001.tar
 |     ├── vid_10001.npy
 |     ├── vid_10001.txt
 |     ├── vid_10001.json
 │     ...
 ...
```

## Example of preparing dataset:

As an example I'll show how we created the CLIP-WebVid dataset:

1. Download the csv files with the video locations and metadata from https://github.com/m-bain/webvid
2. Inspect the csv and adjust to our expected format. If we look at the raw csv's we can see "videoid", "contentUrl", "name", and a few other columns. Well those 3 fit well into our "videoID", "videoLoc", and "caption" descriptions so using pandas we can make that adjustment and save the csv to parquet:
```python3
>>> import pandas as pd
>>> df = pd.read_csv("results_10M_val.csv")
>>> c = df.columns.to_list()
>>> c
['videoid', 'contentUrl', 'duration', 'page_dir', 'name']
>>> c = ["videoID", "videoLoc", "duration", "page_dir", "caption"]
>>> df.columns = c
>>> df
         videoID  ...                                            caption
0     1023443014  ...  Disco light leaks disco ball light reflections...
1     1026421895  ...  Valle de la luna / san pedro de atacama / chil...
2        4350377  ...                   Cloudy moscow kremlin time lapse
3     1054633538  ...           Sharp knife to cut delicious smoked fish
4       15702931  ...  The girl received flowers as a gift. a gift fo...
...          ...  ...                                                ...
4995    16358530  ...  Happy girl sitting in the park and talking on ...
4996    19787275  ...   Young happy family of four on picnic in the park
4997  1021506856  ...  Abstract geometric background texture, geometr...
4998  1047640624  ...  The process of frying and stewing traditional ...
4999    22336669  ...  Lofoten is an archipelago in the county of nor...

[5000 rows x 5 columns]
>>> df.to_parquet("results_10M_val.parquet")
```
3. Now that we have the correct format, clip-video-encode can do the rest. Running the following command will process WebVid (in this case just the validation split) into our target EmbeddingWebDataset format:
```console
clip-video-encode results_10M_val.parquet \
        --dest="my_destination_folder/CLIP_WebVid/val/" \
        --output-format="webdataset" \
        --take-every-nth=25 \
        --frame-workers=48 \
        --frame-memory-size=8 \
        --metadata-columns="videoLoc,caption,duration,page_dir" \
```

## Example of prepared Embedding WebDatasets:
Examples: 
* https://huggingface.co/datasets/iejMac/CLIP-Kinetics700
* https://huggingface.co/datasets/iejMac/CLIP-WebVid
