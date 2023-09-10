"""simplemapper - simple frame -> embedding mapper."""
import torch
import open_clip

from torchvision.transforms import ToPILImage


class FrameMapper:
    """maps frames -> embeddings (or captions"""

    def __init__(self, model_name, pretrained, device, get_tokenizer=False):
        # Initialize model:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(oc_model_name) if get_tokenizer else None
        preprocess.transforms = [ToPILImage()] + preprocess.transforms[-3:]

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

    def generate_captions(self, batch):
        """generate caption for batch of imgs"""
        # TODO: idk if this is the best way to do it but works for now

        # jprompt = "a video of "
        prompt = ""
        tok = self.tokenizer(prompt)
        index = torch.argmax((tok == 49407).type(torch.int64))
        tok = tok[:, :index]
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
