'''
The example below is showcasing how to read an Embedding WebDataset uing the
EmbeddingWebDatasetReader object. 

In order to follow along, below are instructions to download the dataset used
in this example.

In a directory of your choice, from the command line call:

git clone https://huggingface.co/datasets/iejMac/CLIP-MSR-VTT

Next, change directory into the newly created CLIP-MSR-VTT/ and call:

git lfs pull

This will load CLIP encodings of the MSR-VTT dataset onto your machine and
allow you to load that data with the EmbeddingWebDatasetReader.
'''

from clip_video_encode.dataset import EmbeddingWebDatasetReader

val_urls = 'CLIP-MSR-VTT/data/oai_b32/test_full_fps/{000000000..000000007}.tar' #  path to multiple TAR files, the {} notation allows us to specify the range of TAR files we want
val_reader = EmbeddingWebDatasetReader(
    val_urls,
    standard_seq_len=-1,
    batch_size=1,
    num_prepro_workers=2,
    to_tensor=False,
    enable_text=True,
    enable_meta=True
)

#  This newly created val_reader is an iterable, so we can iterate through it
for batch in val_reader:
    #  Print out information about our batch
    print('=====')
    print(batch.keys())
    print(batch['embeddings'].shape)
    print(batch['text'])
    print(batch['meta'])
    print('=====')