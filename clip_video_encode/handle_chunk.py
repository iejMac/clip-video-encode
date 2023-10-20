"""encode chunk with CLIP"""
import numpy as np
import torch

from .utils import block2dl


# BATCH_SIZE = 256
BATCH_SIZE = 128
N_DATASET_WORKERS = 6


def encode_chunk(
    frames,
    ind_dict,
    writer,
    mapper,
    meta,
    ids,
    use_dst_name,
    device,
    input_format="table",
    captioning_strategy="none",
    frame_tokenization_strategy="none",
    generated_caption_key="generated_caption",
):
    """encodes a chunk of video frames and saves."""
    vid_block = np.concatenate(frames)
    dl = block2dl(vid_block, mapper.preprocess, BATCH_SIZE, N_DATASET_WORKERS)

    with torch.no_grad():
        if captioning_strategy != "none":
            captions = []
            for batch in dl:
                captions += mapper.generate_captions(batch.to(device))

            for ref, (i0, it, dst_name) in ind_dict.items():
                vid_id = dst_name[:-4] if use_dst_name else ids[ref]
                if input_format == "webdataset":
                    vid_meta = meta[ref]
                else: # TODO: CHANGE NEEDS TO HAVE EXTENSIONS
                    vid_meta = {}
                    for k in meta:
                        vid_meta[k] = meta[k][ref].as_py()

                # NOTE: Warning this might overwrite previous caption
                # NOTE: for now assumes there is only one caption
                vid_meta[generated_caption_key] = captions[i0:it][0]

                # TODO: we should be able to do both at once with a CoCa model
                writer.write(None, vid_id, vid_meta)
        elif frame_tokenization_strategy != "none":
            tokens = []
            for batch in dl:
                batch = batch.permute(0, 3, 1, 2).float() / 255.0  # make channel first and [0, 1]
                indices = mapper.tokenize_frames(batch.to(device))
                tokens.append(indices)

            tokens = np.concatenate(tokens)

            for ref, (i0, it, dst_name) in ind_dict.items():
                vid_id = dst_name[:-4] if use_dst_name else ids[ref]
                if input_format == "webdataset":
                    vid_meta = meta[ref]
                else:
                    vid_meta = {}
                    for k in meta:
                        vid_meta[k] = meta[k][ref].as_py()

                video_tokens = tokens[i0:it]
                writer.write(video_tokens, vid_id, vid_meta)
        else:
            embeddings = []
            for batch in dl:
                with torch.cuda.amp.autocast():
                    emb = mapper(batch.to(device))
                    embeddings.append(emb)

            caption_embs = None
            if mapper.tokenizer is not None:
                # TODO: is there a better way of doing this?
                # here we will compute similarity of empty string...
                captions = [m["caption"] if "caption" in m else "" for m in meta]
                caption_embs = mapper.encode_captions(captions)
                caption_embs = caption_embs / np.linalg.norm(caption_embs, axis=-1)[:, None]

            embeddings = np.concatenate(embeddings)
            for ref, (i0, it, dst_name) in ind_dict.items():
                vid_id = dst_name[:-4] if use_dst_name else ids[ref]
                if input_format == "webdataset":
                    vid_meta = meta[ref]
                else:
                    vid_meta = {}
                    for k in meta:
                        vid_meta[k] = meta[k][ref].as_py()

                frame_embeddings = embeddings[i0:it]
                if caption_embs is not None:
                    # normalize
                    fe = frame_embeddings / np.linalg.norm(frame_embeddings, axis=-1)[:, None]
                    ce = caption_embs[ref]

                    sim = (fe @ ce.T).tolist()

                    vid_meta["clip_frame_similarity"] = sim

                writer.write(frame_embeddings, vid_id, vid_meta)
