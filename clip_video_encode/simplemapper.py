"""simplemapper - simple frame -> embedding mapper."""
import torch

import open_clip


class FrameMapper:
    """maps frames -> embeddings (or captions"""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, batch):
        with torch.no_grad():
            embeddings = self.model.encode_image(batch).cpu().detach().numpy()
        return embeddings

    def generate_captions(self, batch):
        # TODO: idk if this is the best way to do it but works for now
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = self.model.generate(batch)
        captions = [open_clip.decode(gen).split("<end_of_text>")[0].replace("<start_of_text>", "") for gen in generated]
        return captions
