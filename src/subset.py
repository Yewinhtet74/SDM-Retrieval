"""Fixed random training subsets: draw once with a seed, save to a file,
and reuse the same file on every later run (and from every script).

Single responsibility: choosing which training images/captions to use.

Files are plain text, one image index per line:
    {output_root}/subsets/train_idx_n{size}_seed{seed}.txt
Delete a file to draw a new subset for that size/seed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import CAPS_PER_IMAGE


def subset_indices(n_total, size, seed, out_dir="outputs/subsets"):
    """Return sorted indices of `size` randomly chosen images (no repeats).

    First call with a given (size, seed): draws the indices and saves them.
    Later calls: loads the saved file, so the subset never changes.
    """
    size = min(int(size), int(n_total))
    path = Path(out_dir) / f"train_idx_n{size}_seed{seed}.txt"

    if path.exists():
        idx = np.loadtxt(path, dtype=np.int64, ndmin=1)

        if len(idx) != size or idx.min() < 0 or idx.max() >= n_total:
            raise ValueError(
                f"{path} does not fit the current features "
                f"({len(idx)} indices, max {idx.max()}, n_total={n_total}). "
                f"Delete it to draw a new subset."
            )

        print(f"Loaded {len(idx)} subset indices from {path}")
        return idx

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_total, size=size, replace=False))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, idx, fmt="%d")

    print(f"Saved {len(idx)} new subset indices to {path}")
    return idx


def caption_indices(img_idx, caps_per_image=CAPS_PER_IMAGE):
    """Caption-feature rows that belong to the given images.

    Caption features are stored image by image (see data.TextDataset):
    image i owns rows i*5 ... i*5+4. Taking exactly these rows keeps
    every image paired with its own 5 captions.
    """
    img_idx = np.asarray(img_idx, dtype=np.int64)
    return (img_idx[:, None] * caps_per_image + np.arange(caps_per_image)).ravel()


def train_subset(tr_img, tr_txt, size, seed, out_dir="outputs/subsets"):
    """Pick the same random image subset every time, plus its captions.

    Returns:
        (img_subset, txt_subset, img_idx)
    """
    if len(tr_txt) != len(tr_img) * CAPS_PER_IMAGE:
        raise ValueError(
            f"Expected {CAPS_PER_IMAGE} captions per image, got "
            f"{len(tr_img)} images / {len(tr_txt)} captions"
        )

    img_idx = subset_indices(len(tr_img), size, seed, out_dir)
    txt_idx = caption_indices(img_idx)

    return tr_img[img_idx], tr_txt[txt_idx], img_idx
