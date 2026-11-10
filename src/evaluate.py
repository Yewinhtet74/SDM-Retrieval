"""Evaluation orchestration: given a reducer, evaluate all dimensions -> quality (metrics)
+ engineering (retrieval) -> aggregate results into a result table. Single responsibility:
evaluation orchestration.

Four groups (truncate / pca / adapter_plain / adapter_mrl) use the **same code path**.
The only difference is the reducer passed in — this prevents the claim that "MRL wins"
from being an artifact of different evaluation paths.

Quality is measured on the 1,000-image test set (comparable with the literature);
latency/storage are measured on the full 31k-image gallery (1,000 images are too small,
so latency at every dimension would be sub-millisecond).
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from .features import load_features, load_gallery, model_tag  # model_tag re-exported
from .metrics import recall_at_k, _owner_index
from .retrieval import bench

import numpy as np


def _results_dir(cfg) -> Path:
    d = Path(cfg.paths.output_root) / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _result_path(cfg, stem) -> Path:
    """outputs/results/{model}_{stem}.csv"""
    return _results_dir(cfg) / f"{model_tag(cfg)}_{stem}.csv"



def evaluate_condition(cfg, reducer, name, split="test", data_size=None, with_bench=True):
    """Evaluate each m in cfg.mrl.dims, aggregate results, and save to
    outputs/results/{model}_{name}_{data_size}.csv
    (e.g. ViT-B-32_openai_pca_l2_1000.csv; the model also goes in a "model" column).

    Rows = dimensions;
    columns = bidirectional R@1/5/10 + retention relative to full dimension
    + latency/storage for each precision.
    """
    img, img_ids = load_features(cfg, split, "image")
    txt, txt_ids = load_features(cfg, split, "text")
    gallery, _ = load_gallery(cfg) if with_bench else (None, None)
    # print("img_ids:", img_ids[:20])
    # print("txt_ids:", txt_ids[:30])
    # print("unique image IDs:", len(np.unique(img_ids)))
    # print("unique text IDs:", len(np.unique(txt_ids)))
    dims = list(cfg.mrl.dims)
    tag = model_tag(cfg)

    
    print(f"[{name}] {split}: {len(img_ids)} images / {len(txt_ids)} captions"
          + (f"; gallery {len(gallery)} images" if with_bench else "")
          + f"; dimensions {dims}")

    rows = []
    for m in dims:
        row = {"model": tag, "condition": name, "dim": m}

        # ------------------------------------------------------------
        # DEBUG: verify image/text feature-ID alignment
        # ------------------------------------------------------------
        # print("\n=== RETRIEVAL ALIGNMENT DEBUG ===")

        # owner = _owner_index(img_ids, txt_ids)

        # print("img shape:", img.shape)
        # print("txt shape:", txt.shape)

        # print("first image IDs:")
        # for i in range(min(10, len(img_ids))):
        #     print(i, img_ids[i])

        # print("\nfirst text IDs:")
        # for i in range(min(15, len(txt_ids))):
        #     print(i, txt_ids[i], "owner =", owner[i])

        # # Check caption grouping
        # unique_owner, counts = np.unique(owner, return_counts=True)
        # print("\ncaption count per image:")
        # print("min =", counts.min())
        # print("max =", counts.max())
        # print("mean =", counts.mean())
        # print("images with != 5 captions =", np.sum(counts != 5))

        # # ------------------------------------------------------------
        # # Check whether the correct image/caption pairs have high score
        # # ------------------------------------------------------------
        # img_full = reducer.transform(img, max(dims))
        # txt_full = reducer.transform(txt, max(dims))

        # sim = img_full @ txt_full.T

        # print("\nCorrect-pair similarity sanity check:")

        # correct_scores = []

        # for i in range(min(10, len(img_ids))):
        #     caption_idx = np.where(owner == i)[0]
        #     scores = sim[i, caption_idx]

        #     correct_scores.extend(scores.tolist())

        #     print(
        #         f"image {i} ({img_ids[i]}): "
        #         f"correct captions = {caption_idx.tolist()}, "
        #         f"scores = {scores.tolist()}, "
        #         f"best rank = {1 + np.sum(sim[i] > scores.max())}"
        #     )

        # print(
        #     "\ncorrect-pair score:",
        #     np.mean(correct_scores),
        #     "min:",
        #     np.min(correct_scores),
        #     "max:",
        #     np.max(correct_scores),
        # )

        # print("global similarity:",
        #     sim.min(), sim.mean(), sim.max())

        # print("=== END DEBUG ===\n")
        rec = recall_at_k(
            reducer.transform(img, m),
            reducer.transform(txt, m),
            img_ids,
            txt_ids,
            ks=(1, 5, 10)
        )

        for d in ("i2t", "t2i"):
            for k in (1, 5, 10):
                row[f"{d}_r{k}"] = rec[d][k]

        if with_bench:
            # query = caption (text-to-image retrieval), gallery = 31k images,
            # matching the t2i direction.
            gal_m, q_m = reducer.transform(gallery, m), reducer.transform(txt, m)

            for prec in cfg.eval.precisions:
                b = bench(
                    gal_m,
                    q_m,
                    m,
                    precision=prec,
                    index=cfg.eval.faiss_index
                )
                row[f"lat_ms_{prec}"] = b["lat_ms"]
                row[f"bytes_{prec}"] = b["bytes_per_vec"]
                row[f"index_mb_{prec}"] = b["index_mb"]

            row["n_gallery"] = b["n_gallery"]

        rows.append(row)

        print(
            f"  m={m:>4}  "
            f"i2t R@1={row['i2t_r1']:6.2f}  "
            f"t2i R@1={row['t2i_r1']:6.2f}"
            + (
                f"  lat={row[f'lat_ms_{cfg.eval.precisions[0]}']:6.3f}ms"
                if with_bench else ""
            )
        )

    df = pd.DataFrame(rows)

    full = max(dims)

    # Retention relative to the full dimension — the headline metric.
    for col in ("i2t_r1", "t2i_r1"):
        base = df.loc[df["dim"] == full, col].iloc[0]
        df[f"{col}_ret"] = (
            100.0 * df[col] / base
            if base
            else float("nan")
        )

    out = _result_path(cfg, f"{name}_{data_size}")
    df.to_csv(out, index=False, encoding="utf-8")

    print(f"  -> {out}")
    return df


def load_condition_results(cfg, names):
    """Load the result CSVs for multiple conditions and concatenate them
    vertically (used by 05_summary).

    `names` are condition + training size WITHOUT the model prefix, e.g.
    ["baseline_None", "pca_l2_1000", "svd_l2_1000"]; the current config's model
    prefix is added automatically. Files saved before the prefix existed
    ("pca_l2_1000.csv") are used as a fallback. Missing groups are skipped
    with a warning.
    """
    frames = []

    for n in names:
        p = _result_path(cfg, n)
        if not p.exists() and (_results_dir(cfg) / f"{n}.csv").exists():
            p = _results_dir(cfg) / f"{n}.csv"   # saved before the model prefix

        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            print(f"  (skip {n}: {p.name} does not exist)")

    if not frames:
        raise FileNotFoundError(
            "No result groups found. Run 03 / 04 first."
        )

    return pd.concat(frames, ignore_index=True)