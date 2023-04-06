"""simplemapper - simple frame -> embedding mapper."""
import torch

import open_clip


class FrameMapper:
    """maps frames -> embeddings (or captions"""

    def __init__(self, model, device, tokenizer=None):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer

    def __call__(self, batch, captions=None):
        with torch.no_grad(), torch.cuda.amp.autocast():
            embeddings = self.model.encode_image(batch).cpu().detach().numpy()
        return embeddings

    def encode_captions(self, captions):
        with torch.no_grad(), torch.cuda.amp.autocast():
            tokens = self.tokenizer(captions).to(self.device)
            caption_embeddings = self.model.encode_text(tokens).cpu().detach().numpy()
        return caption_embeddings

    def generate_captions(self, batch):
        # TODO: idk if this is the best way to do it but works for now
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = self.model.generate(batch)
        captions = [open_clip.decode(gen).split("<end_of_text>")[0].replace("<start_of_text>", "") for gen in generated]
        return captions
