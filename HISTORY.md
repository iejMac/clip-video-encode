## 1.3.0

* Transition from OpenAI CLIP to open_clip CLIP + option to select between available CLIP models
* Bug fix - correct video-text alignment for large dataset creation tasks
* Distributing jobs across multiple Slurm nodes is now possible

## 1.2.0

* Take parquet or csv as input and write metadata and captions
* Option to write as EmbeddingWebDataset directly
* inter-video batching
* Use fsspec to support other file systems
* LiveNumpyEncoder for out-of-memory processing from multiple CPU workers

## 1.1.0

* adding dataset section for EmbeddingWebDataset related functionality
* LiveNumpyEncoder - watches directory for incoming numpy frame arrays and encodes them with CLIP
* clip-video-encode now uses video2numpy as video reading/decoding backend

## 1.0.0

* it works
