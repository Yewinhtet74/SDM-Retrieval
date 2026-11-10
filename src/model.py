"""Model: load frozen OpenCLIP and expose preprocess/tokenizer/encoding/logit_scale.
Single responsibility: model + its preprocessing.
"""
from __future__ import annotations

import torch
import open_clip


def load_clip(cfg):
    """Return (model, preprocess, tokenizer).

    The model is set to eval mode, frozen with requires_grad_(False),
    and moved to cfg.model.device.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        device=cfg.model.device,
        precision=cfg.model.precision,
    )

    tokenizer = open_clip.get_tokenizer(cfg.model.name)

    model.eval()
    model.requires_grad_(False)

    return model, preprocess, tokenizer


def encode(model, images=None, texts=None):
    """Encode images or texts using the frozen model.

    Returns unnormalized 512-dimensional embeddings.
    Normalization is intentionally performed at the point of use.
    """
    with torch.no_grad():
        if images is not None:
            img_emb = model.encode_image(images)
            return img_emb

        if texts is not None:
            txt_emb = model.encode_text(texts)
            return txt_emb

        return None