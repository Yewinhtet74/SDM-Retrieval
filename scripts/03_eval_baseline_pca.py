"""Step 3: Baseline (native truncation) + PCA — generate curves with zero training.

Examples
--------
    python scripts/03_baseline_pca.py --train_size 1000 --diagnose     # i2t diagnostic only
    python scripts/03_baseline_pca.py --train_size 1000                # centered PCA  -> pca_1000.csv
    python scripts/03_baseline_pca.py --train_size 1000 --no-center    # uncentered SVD -> svd_1000.csv
    python scripts/03_baseline_pca.py --train_size 1000 --whiten       # PCA + whitening -> pca_whiten_1000.csv
    python scripts/03_baseline_pca.py --train_size 1000 --l2 --diagnose  # diagnostic on L2-normalized features
    python scripts/03_baseline_pca.py --train_size 1000 --l2           # PCA on L2-normalized features -> pca_l2_1000.csv
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evaluate import evaluate_condition
from src.features import load_features
from src.reducers import PCAReducer, TruncateReducer
from src.subset import train_subset


def _l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def main(args):
    cfg = load_config()

    # Training features only (images + text combined, one shared projection).
    # Never use the test set for fitting — prevents data leakage.
    tr_img, _ = load_features(cfg, "train", "image")
    tr_txt, _ = load_features(cfg, "train", "text")
    print(tr_img.shape, tr_txt.shape)
    print(cfg.mrl.dims)

    # Random training subset (same images every run for the same size + seed;
    # indices are saved under outputs/subsets/ and reloaded next time)
    img_sub, txt_sub, _ = train_subset(
        tr_img,
        tr_txt,
        args.train_size,
        args.seed,
        out_dir=pathlib.Path(cfg.paths.output_root) / "subsets",
    )
    print(f"Image size: {img_sub.shape}, Text size: {txt_sub.shape}  (seed={args.seed})")

    # Diagnostic: why is image->text low? Prints a table and exits.
    if args.diagnose:
        from src.diagnose_i2t import diagnose
        from src.metrics import _owner_index

        te_img, img_ids = load_features(cfg, "test", "image")
        te_txt, txt_ids = load_features(cfg, "test", "text")
        arrays = (img_sub, txt_sub, te_img, te_txt)
        if args.l2:
            arrays = tuple(_l2(a) for a in arrays)
            print("(diagnosing L2-normalized features)")
        diagnose(*arrays, _owner_index(img_ids, txt_ids))
        return

    # (1) Native truncation baseline
    # evaluate_condition(cfg, TruncateReducer(), "baseline")

    # (2) PCA / SVD. Each variant gets its own result name so runs don't
    # overwrite each other: pca, svd, pca_whiten, pca_l2, svd_l2, ...
    name = (("pca" if args.center else "svd")
            + ("_whiten" if args.whiten else "")
            + ("_l2" if args.l2 else ""))
    reducer = PCAReducer(
        n_components=tr_img.shape[1],
        center=args.center,
        whiten=args.whiten,
        l2_input=args.l2,   # normalizes train, test and gallery rows alike
    ).fit(img_sub, txt_sub)

    evaluate_condition(cfg, reducer, name, data_size=args.train_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--center", action=argparse.BooleanOptionalAction, default=True,
                        help="--center: true PCA (default). --no-center: uncentered SVD.")
    parser.add_argument("--whiten", action=argparse.BooleanOptionalAction, default=False,
                        help="Divide each component by its std (like sklearn PCA(whiten=True)).")
    parser.add_argument("--l2", action=argparse.BooleanOptionalAction, default=False,
                        help="L2-normalize every feature row before fitting / projecting.")
    parser.add_argument("--diagnose", action="store_true",
                        help="Only run the image->text diagnostic, then exit.")
    args = parser.parse_args()

    print(args)
    main(args)