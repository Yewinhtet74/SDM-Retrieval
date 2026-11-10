"""How much does R@1 drop when the test set is as large as the paper's (~6,358 images)?

Uses the raw embeddings only (no reducer is fitted), so any Flickr30k image can be
used, including training images. Compares a random 1,000-image set (like the
standard test) with a random 6,358-image set (20% of 31,788 images, as in the
paper), each with all 5 captions per image and the usual protocol: i2t is a hit
if any of the image's 5 captions is retrieved, t2i if its one image is retrieved.

    python scripts/check_test_size.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.features import load_features
from src.metrics import recall_at_k

SIZES = (1000, 6358)
SEEDS = (0, 1, 2)


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def main():
    cfg = load_config()
    print(f"model: {cfg.model.name} / {getattr(cfg.model, 'pretrained', '?')}")

    imgs, img_ids, txts, txt_ids = [], [], [], []
    for split in ("train", "val", "test"):
        try:
            ie, ii = load_features(cfg, split, "image")
            te, ti = load_features(cfg, split, "text")
        except (FileNotFoundError, KeyError):
            print(f"  (skip {split}: features not cached)")
            continue
        imgs.append(ie); img_ids += list(ii)
        txts.append(te); txt_ids += list(ti)
    img, txt = l2(np.concatenate(imgs)), l2(np.concatenate(txts))
    print(f"pool: {len(img_ids)} images, {len(txt_ids)} captions")

    rows_of = {}  # image id -> its caption rows
    for k, tid in enumerate(txt_ids):
        rows_of.setdefault(tid.rsplit("_cap", 1)[0], []).append(k)
    usable = [i for i, iid in enumerate(img_ids) if iid in rows_of]

    for n in SIZES:
        res = []
        for seed in SEEDS:
            pick = np.random.default_rng(seed).choice(usable, n, replace=False)
            sel_ids = [img_ids[i] for i in pick]
            cap = [k for iid in sel_ids for k in rows_of[iid]]
            r = recall_at_k(img[pick], txt[cap], sel_ids, [txt_ids[k] for k in cap], ks=(1, 5))
            res.append((r["i2t"][1], r["t2i"][1], r["i2t"][5], r["t2i"][5]))
        m = np.mean(res, axis=0)
        print(f"{n:5d} images / {len(cap):5d} captions:  "
              f"i2t R@1 {m[0]:5.1f}  t2i R@1 {m[1]:5.1f}   "
              f"i2t R@5 {m[2]:5.1f}  t2i R@5 {m[3]:5.1f}   (mean of {len(SEEDS)} random sets)")
    print("paper (6,358 images, OpenAI ViT-B/32): i2t R@1 21.3  t2i R@1 16.8   "
          "i2t R@5 43.8  t2i R@5 40.2")


if __name__ == "__main__":
    main()
