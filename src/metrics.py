"""Quality metrics: bidirectional R@1/5/10 (i2t + t2i). Single responsibility: recall computation.

Protocol (consistent with the literature, 1,000-image test set):
  t2i — each caption has 1 correct image; a hit counts as correct.
  i2t — each image has 5 correct captions; a hit occurs if any correct caption
        appears in the top-K results.
Chunked argpartition is used to avoid materializing the entire similarity matrix.
"""
from __future__ import annotations
import numpy as np

_NORM_TOL = 1e-2


def _check_normalized(emb, name):
    """Input must already be L2-normalized (dot product then equals cosine similarity).

    This also catches the common mistake of normalizing the full embedding first
    and then truncating to a prefix.
    """
    norms = np.linalg.norm(emb[:: max(1, len(emb) // 512)], axis=1)
    if np.abs(norms - 1.0).max() > _NORM_TOL:
        raise ValueError(
            f"{name} is not L2-normalized "
            f"(norm range {norms.min():.4f}~{norms.max():.4f}). "
            f"Note: take the prefix [:m] **before** normalization."
        )


def _owner_index(img_ids, txt_ids):
    """Map txt_id ('{image_id}_cap{j}') to the index of its owner image in img_ids.

    Returns [n_txt] int64.
    """
    pos = {iid: k for k, iid in enumerate(img_ids)}
    owner = np.empty(len(txt_ids), dtype=np.int64)

    for k, tid in enumerate(txt_ids):
        base = tid.rsplit("_cap", 1)[0]
        if base not in pos:
            raise KeyError(
                f"txt_id {tid!r} has owner image {base!r} "
                f"which is not present in img_ids; alignment is broken"
            )
        owner[k] = pos[base]

    return owner


def _topk_desc(query, gallery, k, chunk=2048):
    """Return the top-k gallery indices for each query in descending score order.

    Similarities are computed in chunks to avoid materializing the entire
    query × gallery similarity matrix.

    Returns [n_query, k] int64.
    """
    k = min(k, gallery.shape[0])
    out = np.empty((query.shape[0], k), dtype=np.int64)

    for s in range(0, query.shape[0], chunk):
        sim = query[s:s + chunk] @ gallery.T

        # First roughly select the top-k candidates.
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k]

        # Then sort those candidates by their actual similarity scores.
        rows = np.arange(part.shape[0])[:, None]
        out[s:s + part.shape[0]] = part[
            rows,
            np.argsort(-sim[rows, part], axis=1)
        ]

    return out


def recall_at_k(img_emb, txt_emb, img_ids, txt_ids, ks=(1, 5, 10)):
    """Compute bidirectional R@K.

    img_emb[n_img, m] / txt_emb[n_txt, m] must already be reduced to m
    dimensions and L2-normalized.

    Returns:
        {
            'i2t': {1: ..., 5: ..., 10: ...},
            't2i': {1: ..., 5: ..., 10: ...}
        }

    Values are returned as percentages.
    """
    if len(img_emb) != len(img_ids) or len(txt_emb) != len(txt_ids):
        raise ValueError("Embedding and ID lengths do not match; alignment is broken")

    _check_normalized(img_emb, "img_emb")
    _check_normalized(txt_emb, "txt_emb")

    owner = _owner_index(img_ids, txt_ids)
    kmax = max(ks)

    # i2t:
    # For each image, retrieve the top-kmax captions.
    # A hit occurs if any retrieved caption belongs to that image.
    top_txt = _topk_desc(img_emb, txt_emb, kmax)
    hit_i2t = owner[top_txt] == np.arange(len(img_ids))[:, None]

    # t2i:
    # For each caption, retrieve the top-kmax images.
    # A hit occurs if its correct image is retrieved.
    top_img = _topk_desc(txt_emb, img_emb, kmax)
    hit_t2i = top_img == owner[:, None]

    return {
        "i2t": {
            k: float(
                100.0 * hit_i2t[:, :k].any(axis=1).mean()
            )
            for k in ks
        },
        "t2i": {
            k: float(
                100.0 * hit_t2i[:, :k].any(axis=1).mean()
            )
            for k in ks
        },
    }