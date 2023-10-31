"""simplemapper - simple frame -> embedding mapper."""
import torch
import numpy as np
import open_clip
import torchvision.transforms as T

from torchvision.transforms import ToPILImage
from movqgan import get_movqgan_model


def preprocess_vqgan(x):
    x = x.float() / 127.5 - 1
    return x.permute(0, 3, 1, 2)


class FrameMapper:
    """maps frames -> embeddings (or captions"""

    def __init__(self, model_name, pretrained, device, get_text_tokenizer=False, get_frame_tokenizer=False):
        # Initialize model:
        if not get_frame_tokenizer:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            tokenizer = open_clip.get_tokenizer(oc_model_name) if get_text_tokenizer else None
            preprocess.transforms = [ToPILImage()] + preprocess.transforms[-3:]
        else:
            # https://github.com/ai-forever/MoVQGAN
            model = get_movqgan_model(model_name, pretrained=True, device=device)

            preprocess = T.Compose([
                ToPILImage(),

            ])

            preprocess = lambda x: x  # dataloader preprocess
            tokenizer = lambda x: x

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch, captions=None):
        with torch.no_grad(), torch.cuda.amp.autocast():
            embeddings = self.model.encode_image(batch).cpu().detach().numpy()
        return embeddings

    def encode_captions(self, captions):
        with torch.no_grad(), torch.cuda.amp.autocast():
            tokens = self.tokenizer(captions).to(self.device)
            caption_embeddings = self.model.encode_text(tokens).cpu().detach().numpy()
        return caption_embeddings

    def tokenize_frames(self, batch):
        with torch.no_grad():
            batch = preprocess_vqgan(batch)
            z, _, [_, _, indices] = self.model.encode(batch)
        return indices.reshape(batch.shape[0], -1).cpu().detach().numpy()

    def generate_captions(self, batch):
        """generate caption for batch of imgs"""
        # TODO: idk if this is the best way to do it but works for now

        # jprompt = "a video of "
        prompt = ""
        tok = self.tokenizer(prompt)
        index = torch.argmax((tok == 49407).type(torch.int64))
        tok = tok[:, :index]  # pylint: disable=(invalid-sequence-index)
        tok = torch.cat([tok] * batch.shape[0])
        tok = tok.to(batch.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = self.model.generate(
                batch,
                text=tok,
                generation_type="beam_search",
                temperature=1.0,
                top_p=0.1,
                min_seq_len=15,
                num_beams=10,
                num_beam_groups=5,
            )
        captions = [
            open_clip.decode(gen).split("<end_of_text>")[0].replace("<start_of_text>", "")[len(prompt) :]
            for gen in generated
        ]
        return captions
