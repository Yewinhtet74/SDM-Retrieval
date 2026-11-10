"""Step 1: Sanity check — run frozen CLIP zero-shot retrieval benchmark
to verify official R@K results.

If the results do not match, fix the preprocess/protocol first;
do not proceed to later steps.

    python scripts/01_sanity_check.py                  # Flickr30k 1000-image test
    python scripts/01_sanity_check.py --dataset coco   # MS-COCO 5000-image test (run 09 first)

Also caches the features for this split
(using the same batch-processing path as Step 2).
"""

import argparse
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.features import extract_and_cache, load_features
from src.metrics import recall_at_k


# Official OpenCLIP retrieval results (ViT-B-16 / laion2b_s34b_b88k)
# In docs/openclip_retrieval_results.csv:
# "image retr." = t2i, "text retr." = i2t
# (This naming is counterintuitive, so do not mix them up.)
BENCHMARKS = {
    "flickr30k": {
        "split": "test",
        "desc": "Flickr30k 1000-image test",
        "targets": {
            "i2t": {
                1: 86.30,
                5: 97.90,
                10: 99.40
            },
            "t2i": {
                1: 69.84,
                5: 90.38,
                10: 94.56
            },
        },
    },

    "coco": {
        "split": "coco_test",
        "desc": "MS-COCO 5000-image test",
        "targets": {
            "i2t": {
                1: 59.44,
                5: 81.78,
                10: 88.62
            },
            "t2i": {
                1: 42.31,
                5: 67.70,
                10: 77.06
            },
        },
    },
}


TOL = 2.0   # Percentage points.
            # Tolerance for implementation differences such as fp16
            # and caption selection.


FAIL_HINTS = [
    "Check whether preprocess uses the transform returned by open_clip together with the model",
    "Check whether the split is the Karpathy test split (1000 images / 5000 captions)",
    "Normalization order: normalize only after taking the prefix [:m]",
    "Check whether i2t counts a hit when any of the top-K captions is correct",
]


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset",
        default="flickr30k",
        choices=list(BENCHMARKS)
    )

    args = ap.parse_args()

    bench = BENCHMARKS[args.dataset]
    split, targets = bench["split"], bench["targets"]

    cfg = load_config()

    print(
        f"[Step 1] sanity check  "
        f"{cfg.model.name} / {cfg.model.pretrained}  "
        f"precision={cfg.model.precision}"
    )

    print(f"         Benchmark: {bench['desc']}")

    extract_and_cache(cfg, splits=(split,))

    img_emb, img_ids = load_features(
        cfg,
        split,
        "image"
    )

    txt_emb, txt_ids = load_features(
        cfg,
        split,
        "text"
    )

    print(
        f"{split}: "
        f"{len(img_ids)} images / "
        f"{len(txt_ids)} captions  "
        f"dim={img_emb.shape[1]}"
    )

    res = recall_at_k(
        l2(img_emb),
        l2(txt_emb),
        img_ids,
        txt_ids,
        ks=(1, 5, 10)
    )

    print(
        f"\n{'Direction':<10}"
        f"{'K':>4}"
        f"{'Measured':>12}"
        f"{'Official':>12}"
        f"{'Diff':>10}"
    )

    worst = 0.0

    for d in ("i2t", "t2i"):
        for k in (1, 5, 10):
            got, want = res[d][k], targets[d][k]

            diff = got - want
            worst = max(worst, abs(diff))

            print(
                f"{d:<10}"
                f"{k:>4}"
                f"{got:>11.2f}%"
                f"{want:>11.2f}%"
                f"{diff:>+9.2f}"
            )

    print()

    if worst <= TOL:
        print(
            f"[PASS] Maximum deviation {worst:.2f} percentage points "
            f"(threshold {TOL}). "
            f"Everything looks correct; you can proceed to Step 2."
        )

    else:
        print(
            f"[FAIL] Maximum deviation {worst:.2f} percentage points "
            f"(threshold {TOL}). Check:"
        )

        for h in FAIL_HINTS:
            print(f"  - {h}")

        sys.exit(1)


if __name__ == "__main__":
    main()