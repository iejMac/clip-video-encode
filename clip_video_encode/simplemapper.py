"""simplemapper - simple frame -> embedding mapper."""
import torch
import numpy as np
import open_clip

from torchvision.transforms import ToPILImage

try:
    from omegaconf import OmegaConf
    from taming.models.vqgan import VQModel, GumbelVQ
except ImportError as e:
    print("Missing imports")


def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x


class FrameMapper:
    """maps frames -> embeddings (or captions"""

    def __init__(self, model_name, pretrained, device, get_text_tokenizer=False, get_frame_tokenizer=False):
        # Initialize model:
        if not get_frame_tokenizer:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
            tokenizer = open_clip.get_tokenizer(oc_model_name) if get_text_tokenizer else None
            preprocess.transforms = [ToPILImage()] + preprocess.transforms[-3:]
        else:
            # TODO: you need to download checkpoints/configs from (https://github.com/CompVis/taming-transformers/tree/master#overview-of-pretrained-models)
            config_path, ckpt_path = model_name, pretrained
            config = load_config(config_path, display=False)
            model = load_vqgan(config, ckpt_path=ckpt_path, is_gumbel=('gumbel' in config_path)).to(device)
            # preprocess = preprocess_vqgan
            preprocess = dataloader_preprocess = lambda x: x
            tokenizer = None

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
        return indices.reshape(-1, np.prod(z.shape[-2:])).cpu().detach().numpy()

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
