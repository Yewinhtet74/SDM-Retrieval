"""Step 3: zero-training reducers in one script — truncation baseline, PCA, SVD, SDM.

Every fitted method uses the same random training subset (images + captions
stacked) and the same evaluate_condition() code path, so the curves are
directly comparable. Each reducer always L2-normalizes its output after
reducing to m dimensions; --l2 also normalizes every input row BEFORE fitting
and projecting.

Methods (--methods, any subset, run in the order given):
    baseline   native truncation, no fitting      -> {model}_baseline_None.csv
    pca        standard (centered) PCA            -> {model}_pca_{size}.csv
    pca_whiten PCA + whitening (not in the default)-> {model}_pca_whiten_{size}.csv
    svd        uncentered truncated SVD           -> {model}_svd_{size}.csv
    sdm        dimension merging (slowest to fit) -> {model}_sdm_{size}.csv
{model} is cfg.model.name + cfg.model.pretrained, e.g. ViT-B-32_openai.
With --l2, the fitted methods get "_l2" in their names (..._pca_l2_1000.csv).
The baseline does not change with --l2, so it keeps its name.

Examples (no comments on the line when using %run in Jupyter)
--------
    python scripts/03_eval_reducers.py --train_size 1000
    python scripts/03_eval_reducers.py --train_size 1000 --l2
    python scripts/03_eval_reducers.py --train_size 1000 --l2 --methods pca svd
    python scripts/03_eval_reducers.py --train_size 1000 --l2 --methods pca_whiten
    python scripts/03_eval_reducers.py --train_size 1000 --l2 --diagnose
"""

import argparse
import gc
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evaluate import evaluate_condition
from src.features import feature_dir, load_features, model_tag
from src.reducers import PCAReducer, SDMReducer, SVDReducer, TruncateReducer
from src.subset import train_subset

METHODS = ("baseline", "pca", "pca_whiten", "svd", "sdm")
DEFAULT_METHODS = ("baseline", "pca", "svd", "sdm")


def _l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def build_reducer(method, cfg, l2):
    if method == "pca":
        return PCAReducer(l2_input=l2)
    if method == "pca_whiten":
        return PCAReducer(whiten=True, l2_input=l2)
    if method == "svd":
        return SVDReducer(l2_input=l2)
    if method == "sdm":
        return SDMReducer(dims=cfg.mrl.dims, l2_input=l2)
    raise ValueError(f"unknown method {method!r}")


def run_diagnostic(cfg, img_sub, txt_sub, l2):
    from src.diagnose_i2t import diagnose
    from src.metrics import _owner_index

    te_img, img_ids = load_features(cfg, "test", "image")
    te_txt, txt_ids = load_features(cfg, "test", "text")
    arrays = (img_sub, txt_sub, te_img, te_txt)
    if l2:
        arrays = tuple(_l2(a) for a in arrays)
        print("(diagnosing L2-normalized features)")
    diagnose(*arrays, _owner_index(img_ids, txt_ids))


def main(args):
    cfg = load_config()
    print(args)
    # Features are cached per model in outputs/features/{model}/
    print(f"model: {getattr(cfg.model, 'name', '?')} / {getattr(cfg.model, 'pretrained', '?')}"
          f"   features: {feature_dir(cfg)}")
    methods = list(dict.fromkeys(args.methods))  # drop duplicates, keep order
    suffix = "_l2" if args.l2 else ""
    print(f"methods: {methods}   dims: {list(cfg.mrl.dims)}   l2: {args.l2}")

    # Training subset: loaded once, shared by every fitted method.
    # Never use the test set for fitting — prevents data leakage.
    img_sub = txt_sub = None
    if args.diagnose or any(m != "baseline" for m in methods):
        tr_img, _ = load_features(cfg, "train", "image")
        tr_txt, _ = load_features(cfg, "train", "text")
        print(f"train features: {tr_img.shape} images, {tr_txt.shape} captions")

        # Same images every run for the same size + seed (indices are saved
        # under outputs/subsets/ and reloaded next time)
        img_sub, txt_sub, _ = train_subset(
            tr_img,
            tr_txt,
            args.train_size,
            args.seed,
            out_dir=pathlib.Path(cfg.paths.output_root) / "subsets",
        )
        del tr_img, tr_txt
        print(f"subset: {img_sub.shape} images, {txt_sub.shape} captions  (seed={args.seed})")

    if args.diagnose:
        run_diagnostic(cfg, img_sub, txt_sub, args.l2)
        return

    for method in methods:
        if method == "baseline":
            print("\n=== baseline ===")
            evaluate_condition(cfg, TruncateReducer(), "baseline")
            continue

        print(f"\n=== {method}{suffix} ===")
        reducer = build_reducer(method, cfg, args.l2)
        if method == "sdm":
            # data_size only names the cluster dump: clusters/clusters_{model}_1000[_l2].txt
            reducer.fit(img_sub, txt_sub,
                        data_size=f"{model_tag(cfg)}_{args.train_size}{suffix}")
        else:
            reducer.fit(img_sub, txt_sub)

        evaluate_condition(cfg, reducer, method + suffix, data_size=args.train_size)
        del reducer
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS),
                        help=f"methods to run, in order (default: {' '.join(DEFAULT_METHODS)})")
    parser.add_argument("--train_size", type=int, default=1,
                        help="number of training images in the fitting subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l2", action=argparse.BooleanOptionalAction, default=False,
                        help="Also L2-normalize every input row before fitting / projecting.")
    parser.add_argument("--diagnose", action="store_true",
                        help="Only run the image->text diagnostic on the subset, then exit.")
    args = parser.parse_args()
    main(args)