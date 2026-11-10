"""Data: local Flickr30k (Karpathy split) loading + separate image/text datasets.

Single responsibility: data I/O.

Data source (Option A, fully local, no network access):
  {data_root}/flickr_annotations_30k.csv
      Karpathy annotations: raw (5 captions) / split / filename

  {data_root}/images/{filename}
      31,014 JPG images

Both the CSV and images are downloaded locally by
scripts/00_fetch_data.py from nlphuji/flickr30k.

Convention:
  records contain ONLY image paths and do not hold PIL objects.
  Decoding all images into memory is not feasible with 16 GB RAM.

The image and text streams have different lengths (1 image <-> 5 captions),
so they are implemented as two separate datasets. Both use shuffle=False,
and their iteration order is exactly the order of dataset.ids.
Therefore, cached features and IDs remain aligned.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader


ANNOTATION_CSV = "flickr_annotations_30k.csv"
IMAGE_SUBDIR = "images"
CAPS_PER_IMAGE = 5

# Official Karpathy split sizes.
# A mismatch indicates that the annotation file has been replaced or corrupted.
SPLIT_SIZES = {
    "train": 29000,
    "val": 1014,
    "test": 1000,
}

# Fixed seed used ONLY for the holdout split.
#
# This intentionally does not use cfg.train.seed because the holdout set is
# considered a property of the dataset rather than a property of a particular
# training run. If the split changed with the training seed, retraining with
# another seed could invalidate the guarantee that these images were never
# used for training.
HOLDOUT_SEED = 12345


def _parse_captions(raw):
    """Parse the CSV `raw` column containing a JSON array of captions.

    Quotes inside captions are escaped by JSON encoding.
    """
    caps = json.loads(raw)

    if not isinstance(caps, list):
        raise ValueError(
            f"Caption field is not a list: {raw[:120]!r}"
        )

    return [str(c) for c in caps]


@lru_cache(maxsize=4)
def _read_annotations(root_str: str):
    """Read the annotation CSV -> {split: records}.

    Uses lru_cache to avoid repeatedly parsing the 13 MB CSV when the
    function is called multiple times.
    """
    root = Path(root_str)
    csv_path = root / ANNOTATION_CSV
    img_dir = root / IMAGE_SUBDIR

    if not csv_path.exists() or not img_dir.is_dir():
        raise FileNotFoundError(
            f"Missing local Flickr30k data:\n"
            f"  {csv_path}\n"
            f"  {img_dir}\n"
            f"Run: python scripts/00_fetch_data.py"
        )

    df = pd.read_csv(csv_path)

    for col in ("raw", "split", "filename"):
        if col not in df.columns:
            raise ValueError(
                f"Annotation CSV is missing column {col!r}. "
                f"Available columns: {list(df.columns)}"
            )

    on_disk = {
        p.name
        for p in img_dir.iterdir()
        if p.is_file()
    }

    missing = [
        f
        for f in df["filename"]
        if f not in on_disk
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images are missing from {img_dir} "
            f"(e.g. {missing[:3]}). "
            f"Image extraction is incomplete. "
            f"Re-run scripts/00_fetch_data.py"
        )

    splits: dict[str, list[dict]] = {}

    for row in df.itertuples(index=False):
        caps = _parse_captions(row.raw)

        if len(caps) != CAPS_PER_IMAGE:
            raise ValueError(
                f"{row.filename} contains {len(caps)} captions; "
                f"expected {CAPS_PER_IMAGE}"
            )

        splits.setdefault(row.split, []).append(
            {
                "image_id": Path(row.filename).stem,
                "image_path": str(img_dir / row.filename),
                "captions": caps,
                "split": row.split,
            }
        )

    # Verify official Karpathy split sizes.
    for name, expected in SPLIT_SIZES.items():
        actual = len(splits.get(name, []))

        if actual != expected:
            raise ValueError(
                f"Split {name}: {actual} images, "
                f"expected {expected}. "
                f"Karpathy split does not match."
            )

    return splits


def get_splits(cfg):
    """Return {'train'/'val'/'test': records}.

    Each record contains:
        image_id
        image_path
        captions (5 captions)
        split
    """
    splits = _read_annotations(
        str(Path(cfg.paths.data_root))
    )

    return {
        k: list(v)
        for k, v in splits.items()
    }


# -------------------------------------------------------------------------
# COCO
# -------------------------------------------------------------------------

COCO_ANNOTATION_CSV = "karpathy_coco.csv"

# Official Karpathy COCO split sizes.
#
# train is entirely in train2014.
# validation/test/restval are entirely in val2014.
# This has been verified empirically.
COCO_SPLIT_SIZES = {
    "train": 82783,
    "validation": 5000,
    "test": 5000,
    "restval": 30504,
}


@lru_cache(maxsize=2)
def _read_coco_annotations(root_str: str):
    """Read Karpathy COCO annotations -> {split: records}.

    The record format is identical to Flickr30k.

    Only splits whose images are already available locally are returned.

    By default, this project downloads only val2014
    (validation + test + restval).

    train2014 is the training pool reserved for Phase 2 LoRA and is not
    downloaded by default. It must not accidentally be treated as a
    distractor pool.
    """
    root = Path(root_str)
    csv_path = root / COCO_ANNOTATION_CSV
    img_dir = root / IMAGE_SUBDIR

    if not csv_path.exists() or not img_dir.is_dir():
        raise FileNotFoundError(
            f"Missing local COCO data:\n"
            f"  {csv_path}\n"
            f"  {img_dir}\n"
            f"Run: python scripts/09_fetch_coco.py"
        )

    df = pd.read_csv(csv_path)

    on_disk = {
        p.name
        for p in img_dir.iterdir()
        if p.is_file()
    }

    splits: dict[str, list[dict]] = {}

    for row in df.itertuples(index=False):
        # Ignore images that have not been downloaded.
        if row.filename not in on_disk:
            continue

        # COCO may contain 6-7 captions for some images.
        # Use the first 5 to keep the protocol consistent with Flickr30k.
        caps = _parse_captions(row.raw)[:CAPS_PER_IMAGE]

        # Exclude the very small number of images with fewer than 5 captions.
        if len(caps) < CAPS_PER_IMAGE:
            continue

        splits.setdefault(row.split, []).append(
            {
                "image_id": f"coco{row.cocoid}",
                "image_path": str(img_dir / row.filename),
                "captions": caps,
                "split": row.split,
            }
        )

    return splits


def get_coco_splits(cfg, splits=("test", "restval")):
    """Return COCO records as {split: records}.

    Raises an error if a requested split has not been downloaded.
    """
    all_splits = _read_coco_annotations(
        str(Path(cfg.paths.coco_root))
    )

    out = {}

    for split in splits:
        if split not in all_splits:
            raise FileNotFoundError(
                f"COCO split {split!r} is not available locally. "
                f"Available splits: {sorted(all_splits)}. "
                f"train is contained in train2014.zip and is not "
                f"downloaded by default."
            )

        out[split] = list(all_splits[split])

    return out


# -------------------------------------------------------------------------
# Holdout split
# -------------------------------------------------------------------------

def train_holdout_indices(n_train, n_holdout):
    """Deterministically split train into train and holdout indices.

    Returns:
        (training_indices, holdout_indices)

    Both arrays are sorted.

    The holdout images do NOT participate in any training. They are used as
    clean same-domain distractors to remove the following confounding factor:

        "The gallery contains images seen during adapter training, so those
         images may become stronger false positives and suppress the
         compared condition."

    The same (n_train, n_holdout) always produces the same split because
    HOLDOUT_SEED is fixed.
    """
    import numpy as np

    if not 0 <= n_holdout < n_train:
        raise ValueError(
            f"holdout_train={n_holdout} must be in "
            f"[0, {n_train})"
        )

    perm = np.random.default_rng(
        HOLDOUT_SEED
    ).permutation(n_train)

    train_idx = np.sort(
        perm[n_holdout:]
    )

    holdout_idx = np.sort(
        perm[:n_holdout]
    )

    return train_idx, holdout_idx


# -------------------------------------------------------------------------
# Image dataset
# -------------------------------------------------------------------------

class ImageDataset(Dataset):
    """Dataset containing one item per image.

    Each image is loaded and immediately transformed into a tensor.

    self.ids corresponds one-to-one with dataset iteration order.
    """

    def __init__(self, records, preprocess):
        self.records = records
        self.preprocess = preprocess
        self.ids = [
            r["image_id"]
            for r in records
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        with Image.open(
            self.records[idx]["image_path"]
        ) as im:
            return self.preprocess(
                im.convert("RGB")
            )


# -------------------------------------------------------------------------
# Text dataset
# -------------------------------------------------------------------------

class TextDataset(Dataset):
    """Dataset containing one item per caption.

    Five captions per image are flattened into a single sequence.

    ID format:
        "{image_id}_cap{j}"
    """

    def __init__(self, records, tokenizer):
        self.tokenizer = tokenizer
        self.texts = []
        self.ids = []

        for r in records:
            for j, cap in enumerate(
                r["captions"]
            ):
                self.texts.append(cap)
                self.ids.append(
                    f"{r['image_id']}_cap{j}"
                )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # [1, 77] -> [77]
        # The default DataLoader collate function will add the batch dimension.
        return self.tokenizer(
            self.texts[idx]
        )[0]


# -------------------------------------------------------------------------
# Paired dataset for LoRA training
# -------------------------------------------------------------------------

class PairDataset(Dataset):
    """Dataset returning:

        (preprocessed image, tokenized caption, record index)

    Used for end-to-end LoRA training.

    `cap_choice[i]` specifies which caption is selected for image i
    during the current epoch.

    Exactly ONE caption is sampled per image per epoch.

    This is important because if two captions belonging to the same image
    appear in the same batch, InfoNCE may incorrectly treat one caption
    as a negative example (a false negative).

    The caller therefore samples cap_choice once per epoch and rebuilds
    the DataLoader so that every worker process receives the current
    caption selection.
    """

    def __init__(
        self,
        records,
        cap_choice,
        preprocess,
        tokenizer,
    ):
        if len(cap_choice) != len(records):
            raise ValueError(
                f"cap_choice length {len(cap_choice)} "
                f"!= records length {len(records)}"
            )

        self.records = records
        self.cap_choice = cap_choice
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        with Image.open(
            rec["image_path"]
        ) as im:
            img = self.preprocess(
                im.convert("RGB")
            )

        cap = rec["captions"][
            int(self.cap_choice[idx])
            % len(rec["captions"])
        ]

        return (
            img,
            self.tokenizer(cap)[0],
            idx,
        )


def build_pair_loader(
    cfg,
    records,
    cap_choice,
    preprocess,
    tokenizer,
    batch_size,
    shuffle=True,
    num_workers=None,
):
    """Build paired DataLoader.

    Returns:
        image + selected caption + record index

    The record index is used to retrieve the corresponding distillation
    target.

    `drop_last=True` avoids having a very small final batch that could
    weaken the contrastive learning signal.
    """
    ds = PairDataset(
        records,
        cap_choice,
        preprocess,
        tokenizer,
    )

    nw = (
        cfg.data.num_workers
        if num_workers is None
        else num_workers
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )


# -------------------------------------------------------------------------
# Feature extraction loader
# -------------------------------------------------------------------------

def build_extract_loader(
    cfg,
    records,
    modality,
    preprocess=None,
    tokenizer=None,
):
    """Build a DataLoader for feature extraction.

    Args:
        modality: either "image" or "text".

    Returns:
        (loader, ids)

    IMPORTANT:
        shuffle=False.

    The DataLoader output order is therefore exactly the same as ids,
    ensuring that cached features remain correctly aligned with IDs.
    """
    if modality == "image":
        if preprocess is None:
            raise ValueError(
                "modality='image' requires preprocess"
            )

        ds = ImageDataset(
            records,
            preprocess,
        )

    elif modality == "text":
        if tokenizer is None:
            raise ValueError(
                "modality='text' requires tokenizer"
            )

        ds = TextDataset(
            records,
            tokenizer,
        )

    else:
        raise ValueError(
            f"modality must be 'image' or 'text', "
            f"got {modality!r}"
        )

    # Tokenization is fast, so spawning workers for text usually adds
    # overhead rather than improving performance.
    nw = (
        cfg.data.num_workers
        if modality == "image"
        else 0
    )

    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size_extract,
        num_workers=nw,
        shuffle=False,
        pin_memory=True,
        persistent_workers=bool(nw),
    )

    return loader, ds.ids