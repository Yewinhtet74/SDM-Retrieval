"""Features: extract and cache frozen CLIP 512-dimensional embeddings (.npy)
and provide loading interfaces. ★ FOUNDATION ★
Single responsibility: feature caching.

Convention: store **raw, unnormalized fp32** embeddings
(normalization is applied when needed; otherwise prefix truncation order
would be fixed prematurely).

Image/text features are stored separately with aligned ID tables.
If a feature file already exists, extraction is skipped.

Each model gets its own cache folder, outputs/features/{model}/, where {model}
is cfg.model.name + cfg.model.pretrained (e.g. ViT-B-32_openai), so features
from different encoders can never be mixed up. A model.json in each folder
records which model produced the files; load_features refuses a folder whose
model.json does not match the config.

.npy is the single source of truth. All subsequent PCA / adapter training /
four-group evaluation runs operate on these cached features.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data import build_extract_loader, get_coco_splits, get_splits
from .model import encode

MODALITIES = ("image", "text")
COCO_PREFIX = "coco_"


def _resolve_splits(cfg, splits):
    """Dispatch datasets based on split names:
    `coco_*` -> MS-COCO, everything else -> Flickr30k.

    This allows `load_features(cfg, "coco_restval", "image")`
    to use exactly the same interface as Flickr30k.

    Higher-level modules such as evaluate / cascade do not need to know
    which dataset the features come from.
    """
    flickr = [s for s in splits if not s.startswith(COCO_PREFIX)]
    coco = [s[len(COCO_PREFIX):] for s in splits if s.startswith(COCO_PREFIX)]

    out = {}

    if flickr:
        out.update({
            k: v
            for k, v in get_splits(cfg).items()
            if k in flickr
        })

    if coco:
        out.update({
            COCO_PREFIX + k: v
            for k, v in get_coco_splits(cfg, coco).items()
        })

    return out


def model_tag(cfg) -> str:
    """File-name-safe model id, e.g. 'ViT-B-32_openai' or
    'ViT-B-32_laion2b_s34b_b79k' (cfg.model.name + cfg.model.pretrained;
    characters like '/' or '@' become '-'). Used for the feature folder and as
    the prefix of result files."""
    name = getattr(cfg.model, "name", "model")
    pretrained = getattr(cfg.model, "pretrained", None)
    tag = f"{name}_{pretrained}" if pretrained else str(name)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tag)


def _feature_root(cfg) -> Path:
    return Path(cfg.paths.output_root) / "features"


def _feature_dir(cfg) -> Path:
    """Return the feature cache directory of the configured model:
    outputs/features/{model}/"""
    d = _feature_root(cfg) / model_tag(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d


feature_dir = _feature_dir  # public name


def _model_desc(cfg) -> str:
    return f"{getattr(cfg.model, 'name', '?')} / {getattr(cfg.model, 'pretrained', '?')}"


def _check_model_json(cfg, d: Path):
    """Raise if the folder's model.json says it holds another model's features."""
    meta = d / "model.json"
    if not meta.exists():
        return
    m = json.loads(meta.read_text(encoding="utf-8"))
    want = (getattr(cfg.model, "name", None), getattr(cfg.model, "pretrained", None))
    if (m.get("name"), m.get("pretrained")) != want:
        raise RuntimeError(
            f"{d} holds features of {m.get('name')} / {m.get('pretrained')}, "
            f"but the config says {want[0]} / {want[1]}."
        )


def _paths(cfg, split, modality):
    """Return the paths for an embedding .npy file and its ID JSON file."""
    d = _feature_dir(cfg)
    return (
        d / f"{split}_{modality}.npy",
        d / f"{split}_{modality}_ids.json"
    )


def _image_input_dtype(model):
    """Determine the dtype expected by the first layer of the visual encoder.

    The image preprocessing pipeline outputs fp32 tensors. If the model uses
    mixed precision, passing fp32 directly may cause a dtype mismatch.

    Important: open_clip with precision='fp16' uses a **mixed-precision model**,
    not a model where every parameter is fp16.

    For example, in open_clip 3.3.0:
      positional_embedding -> fp32
      logit_scale          -> fp32
      visual.conv1.weight  -> fp16

    Therefore, `next(model.parameters()).dtype` can incorrectly report fp32.

    Instead, find the first Conv/Linear layer inside `model.visual`.
    For ViT this is usually `conv1`; for timm backbones it is typically
    `patch_embed.proj`.
    """
    for m in model.visual.modules():
        if isinstance(
            m,
            (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)
        ):
            w = getattr(m, "weight", None)
            if w is not None:
                return w.dtype

    return next(model.visual.parameters()).dtype


@torch.no_grad()
def _encode_loader(
    loader,
    ids,
    model,
    modality,
    device,
    img_dtype,
    desc
):
    """Run one complete DataLoader pass and produce [N, dim] fp32
    unnormalized features.

    The feature order is exactly the same as the ID order because
    the loader uses shuffle=False.
    """
    out = []

    for batch in tqdm(loader, desc=desc, unit="batch"):
        batch = batch.to(device, non_blocking=True)

        if modality == "image":
            emb = encode(
                model,
                images=batch.to(img_dtype)
            )
        else:
            # Tokens are integer tensors and must be passed directly
            # to the embedding layer.
            emb = encode(
                model,
                texts=batch
            )

        out.append(
            emb.float().cpu().numpy()
        )

    emb = np.concatenate(out, axis=0).astype(np.float32)

    if len(emb) != len(ids):
        raise RuntimeError(
            f"{desc}: {len(emb)} features != {len(ids)} IDs; "
            "feature/ID alignment is broken"
        )

    return emb


def extract_and_cache(
    cfg,
    splits=("train", "val", "test"),
    modalities=MODALITIES,
    overwrite=False
):
    """Extract frozen CLIP features for the specified splits/modalities.

    Features are cached as raw fp32 embeddings in:
        outputs/features/{model}/

    `modalities` can contain only ``image`` when text features are unnecessary,
    such as for the COCO restval interference gallery.
    Extracting tens of thousands of unused captions would otherwise waste time.
    """
    from .model import load_clip

    d = _feature_dir(cfg)
    print(f"Model: {_model_desc(cfg)}   feature folder: {d}")

    todo = [
        (s, m)
        for s in splits
        for m in modalities
        if overwrite or not _paths(cfg, s, m)[0].exists()
    ]

    if not todo:
        print(
            f"All features for this model are already cached in {d}. "
            "Skipping extraction. Delete that folder or use overwrite=True "
            "to re-extract."
        )
        return {}

    print(
        f"Features to extract: "
        f"{[f'{s}/{m}' for s, m in todo]}"
    )

    all_splits = _resolve_splits(
        cfg,
        sorted({s for s, _ in todo})
    )

    model, preprocess, tokenizer = load_clip(cfg)

    device = cfg.model.device
    img_dtype = _image_input_dtype(model)

    print(
        f"Image input dtype = {img_dtype} "
        f"(precision={cfg.model.precision})"
    )

    done = {}

    for split, modality in todo:
        records = all_splits[split]

        loader, ids = build_extract_loader(
            cfg,
            records,
            modality,
            preprocess=preprocess,
            tokenizer=tokenizer
        )

        emb = _encode_loader(
            loader,
            ids,
            model,
            modality,
            device,
            img_dtype,
            desc=f"{split}/{modality}"
        )

        emb_path, ids_path = _paths(
            cfg,
            split,
            modality
        )

        np.save(emb_path, emb)

        ids_path.write_text(
            json.dumps(ids),
            encoding="utf-8"
        )

        print(
            f"  {emb_path}  "
            f"{emb.shape}  "
            f"{emb.nbytes / 1e6:.1f} MB"
        )

        done[(split, modality)] = emb.shape

    # Record which model produced the files in this folder
    (d / "model.json").write_text(json.dumps({
        "name": getattr(cfg.model, "name", None),
        "pretrained": getattr(cfg.model, "pretrained", None),
        "precision": getattr(cfg.model, "precision", None),
        "dim": int(emb.shape[1]),
    }, indent=2), encoding="utf-8")

    return done


def load_features(cfg, split, modality):
    """Load cached features.

    Parameters
    ----------
    split : str
        Dataset split name, e.g. ``train``, ``val``, ``test``,
        ``coco_test``, or ``coco_restval``.

    modality : str
        Either ``image`` or ``text``.

    Returns
    -------
    emb : np.ndarray
        Shape [N, 512], float32, unnormalized embeddings.

    ids : list
        IDs aligned with the embedding rows.
    """
    if modality not in MODALITIES:
        raise ValueError(
            f"modality must be image/text, got {modality!r}"
        )

    emb_path, ids_path = _paths(
        cfg,
        split,
        modality
    )

    if not emb_path.exists():
        hint = ""
        legacy = _feature_root(cfg) / emb_path.name
        if legacy.exists():
            hint = (
                f" There is an older cache directly in {legacy.parent} (made "
                "before model folders, so its model is not recorded). If it "
                f"came from {_model_desc(cfg)}, move those files into "
                f"{emb_path.parent}; otherwise re-extract."
            )
        raise FileNotFoundError(
            f"Missing feature cache {emb_path} for {_model_desc(cfg)}. "
            "Run: python scripts/02_extract_features.py." + hint
        )
    _check_model_json(cfg, emb_path.parent)

    emb = np.load(emb_path).astype(np.float32)

    ids = json.loads(
        ids_path.read_text(
            encoding="utf-8"
        )
    )

    return emb, ids


def load_gallery(cfg, splits=("train", "val", "test")):
    """Build the full ~31k-image gallery.

    This gallery is used for latency/storage measurements.
    A 1,000-image gallery is too small: latency at every dimension
    would be sub-millisecond and dimension-dependent differences
    would be difficult to observe.

    Returns
    -------
    emb : np.ndarray
        Shape [N, 512], fp32.

    ids : list
        Image IDs.

    WARNING
    -------
    This gallery contains 29,000 training images that were seen by
    the adapter during training.

    Therefore, this gallery should NOT be used for cross-condition
    recall comparisons.

    See RESULTS.md §5 for the measurement caveat.
    Use `load_clean_gallery()` for a clean gallery.
    """
    embs, ids = [], []

    for s in splits:
        e, i = load_features(
            cfg,
            s,
            "image"
        )

        embs.append(e)
        ids.extend(i)

    return np.concatenate(embs, axis=0), ids


def load_clean_gallery(cfg, kind="same_domain"):
    """Build a **clean** gallery containing no adapter-training images.

    Returns
    -------
    emb : np.ndarray
        Shape [N, 512], fp32.

    ids : list
        Image IDs.

    Parameters
    ----------
    kind : str
        ``same_domain``
            Test 1,000 + held-out Flickr30k training images.
            Same domain, smaller gallery, distributionally consistent.

        ``cross_domain``
            Test 1,000 + COCO restval 30,504.
            Large gallery, but contains domain shift.

            COCO tends to be object-centered, whereas Flickr30k contains
            more people/action-oriented images. Cross-domain distractors
            may therefore be easier to reject.

    Both should be reported so that conclusions are not dependent
    on a particular gallery construction.
    """
    from .data import train_holdout_indices

    te, te_ids = load_features(
        cfg,
        "test",
        "image"
    )

    if kind == "same_domain":
        tr, tr_ids = load_features(
            cfg,
            "train",
            "image"
        )

        _, held = train_holdout_indices(
            len(tr_ids),
            cfg.data.holdout_train
        )

        return (
            np.concatenate(
                [te, tr[held]],
                axis=0
            ),
            list(te_ids)
            + [tr_ids[i] for i in held]
        )

    if kind == "cross_domain":
        co, co_ids = load_features(
            cfg,
            "coco_restval",
            "image"
        )

        return (
            np.concatenate(
                [te, co],
                axis=0
            ),
            list(te_ids) + list(co_ids)
        )

    raise ValueError(
        f"kind must be same_domain/cross_domain, got {kind!r}"
    )