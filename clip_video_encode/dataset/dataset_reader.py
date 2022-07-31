"""
utils for processing datasets of format described in https://github.com/iejMac/clip-video-encode/pull/13

used https://github.com/rom1504/laion-prepro/blob/main/laion5B/usage_guide/dataloader_pytorch.py as template
"""

import io

import clip
import numpy as np
import torch
import webdataset as wds

from torch.utils.data import DataLoader


def standardize_embedding_shape(emb, seq_len):
    if len(emb) > seq_len:
        print(f"Warning: Raw embedding is longer than standard sequence length ({len(emb)} > {seq_len})")
        emb = emb[:seq_len]

    pad = np.zeros((seq_len - len(emb), emb.shape[1]), dtype=emb.dtype)
    zero_mask = np.concatenate([np.ones(len(emb)), np.zeros(len(pad))])
    padded_emb = np.concatenate([emb, pad])
    return padded_emb, zero_mask


def create_embeddingwebdataset(
    urls,
    embedding_transform=lambda emb: emb,
    standard_seq_len=-1,
    to_tensor=True,
    enable_text=True,
    enable_meta=True,
):
    """
    Create a WebDataset reader for Frame Embedding Dataset

    Input:
        standard_seq_len: sequence length to pad all embedding sequences to (for batching)
            !(-1) : pad to standard_seq_len
            -1: don't pad (dataset can't be used in DataLoader with batch_size > 1)
        enable_text: include text captions
        enable_meta: include metadata
    """

    dataset = wds.WebDataset(urls)
    # TODO: different tokeinzers??
    def tokenizer(text):
        return clip.tokenize([text], truncate=True)[0]

    def preprocess_dataset(item):
        output = {}

        npy_data = item["npy"]
        stream = io.BytesIO(npy_data)
        emb = np.lib.format.read_array(stream)

        if standard_seq_len != -1:
            emb, zero_mask = standardize_embedding_shape(emb, standard_seq_len)
            output["zero_mask"] = zero_mask
        if to_tensor:
            emb = torch.from_numpy(emb)

        output["embeddings"] = embedding_transform(emb)

        if enable_text:
            text_data = item["cap"]
            text = text_data.decode("utf-8")
            output["text"] = text
            output["text_tokens"] = tokenizer(text)
        if enable_meta:
            meta_data = item["json"]
            meta = meta_data.decode("utf-8")
            output["meta"] = meta
        return output

    transformed_dataset = dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    return transformed_dataset


def dataset_to_dataloader(dataset, batch_size, num_prepro_workers):
    """converts WebDataset to PyTorch DataLoader."""

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_prepro_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    return dl


class EmbeddingWebDatasetReader:
    """WebDataset reader for Embedding Datasets"""

    def __init__(
        self,
        urls,
        standard_seq_len,
        batch_size,
        num_prepro_workers,
        to_tensor=True,
        enable_text=True,
        enable_meta=False,
        embedding_transform=lambda emb: emb,
    ):
        self.batch_size = batch_size
        dataset = create_embeddingwebdataset(
            urls,
            embedding_transform,
            standard_seq_len,
            to_tensor,
            enable_text,
            enable_meta,
        )
        self.dataloader = dataset_to_dataloader(dataset, batch_size, num_prepro_workers)

    def __iter__(self):
        for batch in self.dataloader:
            yield batch
