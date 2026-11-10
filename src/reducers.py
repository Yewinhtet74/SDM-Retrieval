"""Unified dimensionality-reduction interface (PyTorch version).

Maps D-dim features to m-D retrieval vectors. Reducers in this file:
    TruncateReducer  first m dimensions (baseline)
    PCAReducer       standard (centered) PCA
    SVDReducer       uncentered truncated SVD
    SDMReducer       dimension merging

All reducers share one interface:

    reducer.fit(train_img, train_txt)
    reducer.transform(emb, m)   -> L2-normalized [N, m]

Key conventions (unchanged from the NumPy version):
    * `transform(emb, m)` returns L2-normalized vectors, and normalization
      always happens AFTER truncating / aggregating to m dimensions.
    * With `l2_input=True`, every input row is ALSO L2-normalized BEFORE
      fitting / projecting (training, test and gallery rows alike).
    * `_project` results are memoized by input object, because evaluation
      queries the same `emb` at several dimensions.

PyTorch-specific conventions:
    * All computation runs on `self.device` (cuda if available, else cpu).
    * `transform` accepts a NumPy array OR a torch tensor and returns the same
      kind it was given (NumPy in -> NumPy float32 out). This keeps the existing
      evaluate.py / metrics / faiss code working without changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_tensor(x, device, dtype=torch.float32) -> torch.Tensor:
    """NumPy array / list / tensor -> tensor on `device` with `dtype`."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _to_input_type(out: torch.Tensor, ref):
    """Return `out` as the same kind of object as `ref` (tensor or NumPy)."""
    if isinstance(ref, torch.Tensor):
        return out
    return np.ascontiguousarray(out.detach().cpu().numpy(), dtype=np.float32)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize each row. Zero vectors use norm 1e-12 to avoid 0/0."""
    n = torch.linalg.norm(x, dim=1, keepdim=True)
    return x / n.clamp_min(1e-12)


def numpy_converter(obj):
    """`json.dump(default=...)` hook for NumPy / torch objects."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ----------------------------------------------------------------------------
# Cluster aggregation
# ----------------------------------------------------------------------------
def _cluster_layout(clus_ids: torch.Tensor):
    """Map arbitrary cluster labels to 0..k-1.

    Returns
    -------
    idx : [cols]  compact cluster index of every column
    first_indices : [k]  smallest column index inside each cluster
    counts : [k]  number of columns in each cluster
    """
    cols = clus_ids.numel()
    cluster_ids, idx = torch.unique(clus_ids, return_inverse=True)
    k = cluster_ids.numel()

    first_indices = torch.full((k,), cols, dtype=torch.long, device=clus_ids.device)
    first_indices.scatter_reduce_(
        0,
        idx,
        torch.arange(cols, dtype=torch.long, device=clus_ids.device),
        reduce="amin",
        include_self=True,
    )
    counts = torch.bincount(idx, minlength=k)
    return idx, first_indices, counts


def _cluster_means(X: torch.Tensor, idx, counts, k) -> torch.Tensor:
    """Mean of the columns of each cluster -> [rows, k] (cluster 0..k-1 order)."""
    sums = torch.zeros((X.shape[0], k), dtype=X.dtype, device=X.device)
    sums.index_add_(1, idx, X)
    return sums / counts.to(X.dtype)


def cluster_means(X: torch.Tensor, clus_ids: torch.Tensor):
    """Per-cluster column means, ordered by each cluster's first column index.

    Returns
    -------
    means : [rows, k]   (== aggregate_to_first(X, clus_ids)[:, first_mask])
    first : [k]         first original column index of each returned cluster,
                        in ascending order
    """
    clus_ids = clus_ids.to(X.device)
    idx, first, counts = _cluster_layout(clus_ids)
    means = _cluster_means(X, idx, counts, first.numel())
    order = torch.argsort(first)
    return means[:, order], first[order]


def aggregate_to_first(X: torch.Tensor, clus_ids: torch.Tensor) -> torch.Tensor:
    """Replace each cluster's first column by the cluster mean; fill the rest with inf.

    X: [rows, cols] -> [rows, cols]
    """
    rows, cols = X.shape
    clus_ids = torch.as_tensor(clus_ids, device=X.device)
    idx, first, counts = _cluster_layout(clus_ids)
    means = _cluster_means(X, idx, counts, first.numel())

    out = torch.full((rows, cols), float("inf"), dtype=X.dtype, device=X.device)
    out[:, first] = means
    return out


def first_occurrence_mask(items) -> torch.Tensor:
    """Bool mask [cols] that is True at the first occurrence of every label.

    (The NumPy version returned a list of 0/1; this returns a bool tensor.)
    """
    items = torch.as_tensor(items)
    _, first, _ = _cluster_layout(items)
    mask = torch.zeros(items.numel(), dtype=torch.bool, device=items.device)
    mask[first] = True
    return mask


# ----------------------------------------------------------------------------
# Pairwise distance between columns
# ----------------------------------------------------------------------------
@torch.no_grad()
def finite_avg_distance(
    X: torch.Tensor,
    chunk_size: int = 256,
    acc_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    X: [rows, cols]

    D[i, j] = mean |X[:, i] - X[:, j]| over rows where BOTH values are finite
    (inf where there is no such row). Differences are computed in X.dtype and
    accumulated in `acc_dtype` (float64), exactly like the NumPy version.

    Memory per step is ~ chunk_size * cols * cols elements; lower `chunk_size`
    if the GPU runs out of memory.
    """
    rows, cols = X.shape
    finite = torch.isfinite(X)
    all_finite = bool(finite.all())

    sum_diff = torch.zeros((cols, cols), dtype=acc_dtype, device=X.device)
    count = torch.zeros((cols, cols), dtype=torch.long, device=X.device)

    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        x = X[start:end]

        diff = (x.unsqueeze(2) - x.unsqueeze(1)).abs_()  # [chunk, cols, cols]

        if all_finite:
            count += end - start
        else:
            f = finite[start:end]
            both = f.unsqueeze(2) & f.unsqueeze(1)
            diff.masked_fill_(~both, 0)
            count += both.sum(dim=0)

        sum_diff += diff.sum(dim=0, dtype=acc_dtype)
        del x, diff

    D = torch.full((cols, cols), float("inf"), dtype=acc_dtype, device=X.device)
    valid = count > 0
    D[valid] = sum_diff[valid] / count[valid].to(acc_dtype)
    return D


# ----------------------------------------------------------------------------
# Hierarchical clustering of columns
# ----------------------------------------------------------------------------
def _prepare_cluster_inputs(X, num_clusters):
    if num_clusters is None:
        raise ValueError("num_clusters must be provided")
    num_clusters = list(num_clusters)

    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(np.asarray(X))
    if not X.is_floating_point():
        X = X.float()
    return X, num_clusters


@torch.no_grad()
def calculate_cluster(X, nums=None, num_clusters=None, chunk_size: int = 256):
    """Greedy agglomerative clustering of the COLUMNS of X.

    Repeatedly merges the two clusters whose mean columns have the smallest
    finite average absolute distance, and records the cluster assignment each
    time the cluster count hits one of `num_clusters`.

    Parameters
    ----------
    X : [rows, cols] tensor (or array). Put it on the GPU for speed.
    nums : unused (kept for API compatibility).
    num_clusters : iterable of target cluster counts, e.g. [512, 256, 128, 64].

    Returns
    -------
    dict {n_clusters: LongTensor[cols] of cluster labels (on X.device)}
    """
    X, num_clusters = _prepare_cluster_inputs(X, num_clusters)
    cols = X.shape[1]
    target = min(num_clusters)

    clus_ids = torch.arange(cols, dtype=torch.long, device=X.device)
    all_clusters = {n: None for n in num_clusters}
    all_clusters[cols] = clus_ids.clone()

    k = cols
    while k > target:
        # Cluster-mean columns only (k columns instead of `cols` columns with inf).
        # Ordered by first column index, so the lower-triangle scan order -- and
        # therefore tie-breaking -- is identical to the NumPy version.
        means, first = cluster_means(X, clus_ids)
        D = finite_avg_distance(means, chunk_size)

        row_idx, col_idx = torch.tril_indices(k, k, offset=-1, device=X.device)
        j = torch.argmin(D[row_idx, col_idx])

        old_cluster = clus_ids[first[row_idx[j]]]
        new_cluster = clus_ids[first[col_idx[j]]]
        clus_ids = torch.where(clus_ids == old_cluster, new_cluster, clus_ids)

        k -= 1
        if k in all_clusters:
            all_clusters[k] = clus_ids.clone()

    return all_clusters


@torch.no_grad()
def calculate_cluster_new(
    X,
    nums=None,
    num_clusters=None,
    max_merges_per_step: int = 1,
    chunk_size: int = 256,
    verbose: bool = False,
):
    """Variant of `calculate_cluster` that can merge several disjoint pairs per
    distance computation (faster, slightly different result).

    With `max_merges_per_step=1` this is the same algorithm as `calculate_cluster`
    (the NumPy `calculate_cluster_new` had the limit hard-coded to `min(1, ...)`).

    The number of merges in a step is also capped so the cluster count never
    jumps over an intermediate requested size (otherwise `all_clusters[n]`
    would silently stay None).
    """
    X, num_clusters = _prepare_cluster_inputs(X, num_clusters)
    cols = X.shape[1]
    targets = sorted(set(num_clusters))

    clus_ids = torch.arange(cols, dtype=torch.long, device=X.device)
    all_clusters = {n: None for n in num_clusters}
    all_clusters[cols] = clus_ids.clone()

    k = cols
    while k > targets[0]:
        means, first = cluster_means(X, clus_ids)
        D = finite_avg_distance(means, chunk_size)

        row_idx, col_idx = torch.tril_indices(k, k, offset=-1, device=X.device)
        values = D[row_idx, col_idx]

        next_target = max(t for t in targets if t < k)
        max_merges = max(1, min(max_merges_per_step, k - next_target))

        if max_merges == 1:
            order = torch.argmin(values).view(1)
        else:
            order = torch.argsort(values, stable=True)

        cl_a = clus_ids[first[row_idx[order]]].tolist()
        cl_b = clus_ids[first[col_idx[order]]].tolist()

        selected, used = [], set()
        for a, b in zip(cl_a, cl_b):
            if a in used or b in used:  # avoid overlapping merges
                continue
            selected.append((a, b))
            used.update((a, b))
            if len(selected) == max_merges:
                break

        if verbose:
            print(f"Selected pairs for merging: {selected}")

        for old_cluster, new_cluster in selected:
            clus_ids = torch.where(clus_ids == old_cluster, new_cluster, clus_ids)

        k -= len(selected)
        if k in all_clusters:
            all_clusters[k] = clus_ids.clone()

    return all_clusters


# ----------------------------------------------------------------------------
# Reducers
# ----------------------------------------------------------------------------
class Reducer:
    """Base reducer.

    `fit(train...)` fits the reducer if necessary (may be a no-op).
    `transform(emb, m)` returns an L2-normalized m-dimensional representation.

    Subclasses only need to implement `_project(emb: Tensor) -> Tensor[N, D]`;
    truncation, normalization, memoization and NumPy<->torch conversion are
    handled here.

    l2_input : False (default). True L2-normalizes every INPUT row before the
        reducer sees it: training rows in `fit` and test / gallery rows in
        `transform`, so train and test are always treated the same way. Use it
        when the cached features are not normalized. (Output vectors are
        always L2-normalized, whatever this setting.)
    """

    name = "reducer"
    _MEMO_CAP = 8

    def __init__(self, device=None, l2_input=False):
        self.device = torch.device(device) if device is not None else _default_device()
        self.l2_input = l2_input
        # id(emb) -> (input object, projected tensor on self.device).
        # Keeping a reference to the input lets us verify identity with `is`
        # and stops Python from recycling the id.
        self._memo = {}

    def fit(self, train_img, train_txt):
        return self

    def _input(self, x) -> torch.Tensor:
        """Input rows -> float32 tensor on self.device (L2-normalized if l2_input)."""
        x = _as_tensor(x, self.device)
        return l2_normalize(x) if self.l2_input else x

    def _project(self, emb: torch.Tensor) -> torch.Tensor:
        """Project [N, 512] float32 (already on self.device) to [N, D]."""
        return emb

    def clear_memo(self):
        self._memo.clear()

    @torch.no_grad()
    def transform(self, emb, m):
        """Project, truncate to m dimensions, then L2-normalize."""
        hit = self._memo.get(id(emb))

        if hit is None or hit[0] is not emb:
            if len(self._memo) >= self._MEMO_CAP:
                self._memo.clear()
            proj = self._project(self._input(emb)).float().contiguous()
            hit = (emb, proj)
            self._memo[id(emb)] = hit

        proj = hit[1]

        if m > proj.shape[1]:
            print(
                f"{self.name}: requested {m} dimensions, "
                f"but the representation only has {proj.shape[1]} dimensions"
            )

        # Important: truncate first, then normalize.
        return _to_input_type(l2_normalize(proj[:, :m]), emb)


class TruncateReducer(Reducer):
    """(1) Native truncation: take the first m dimensions, then L2-normalize."""

    name = "truncate"


class _LinearReducer(Reducer):
    """Shared code for PCAReducer and SVDReducer (not meant to be used directly).

    Fits one orthogonal projection on the stacked training images + captions.
    `transform(emb, m)` keeps the first m components, then L2-normalizes.

    Options
    -------
    n_components : number of components to keep (default: all features).
        With solver="lowrank" this is also `q` for `torch.pca_lowrank`.
    solver : "exact" (default) = exact eigendecomposition in float64,
        deterministic. "lowrank" = `torch.pca_lowrank` (randomized, float32;
        call torch.manual_seed(...) before fit() for reproducible results).
    niter : subspace iterations for solver="lowrank" (torch default: 2).
    l2_input : L2-normalize every input row first (see `Reducer`).

    Deterministic sign: the largest-magnitude entry of each component is
    positive (sklearn's svd_flip convention); it does not affect retrieval.
    """

    _center = False  # set by subclasses
    _kind = ""

    def __init__(self, n_components=None, solver="exact", niter=2, l2_input=False,
                 device=None):
        super().__init__(device=device, l2_input=l2_input)
        if solver not in ("lowrank", "exact"):
            raise ValueError(f"solver must be 'lowrank' or 'exact', got {solver!r}")
        self.n_components = n_components
        self.solver = solver
        self.niter = niter
        self.mean_ = None       # [1, D_in] training mean (PCA only)
        self.components = None  # [D_in, k] projection matrix (unit-norm columns)
        self.explained_variance_ratio_ = None  # [k] as in sklearn PCA / TruncatedSVD

    @torch.no_grad()
    def fit(self, train_img, train_txt):
        X = torch.cat([self._input(train_img), self._input(train_txt)], dim=0)
        n, d = X.shape

        k = min(self.n_components or d, n, d)
        if self.n_components is not None and k < self.n_components:
            print(
                f"Warning: requested n_components={self.n_components} exceeds "
                f"min(n_samples={n}, n_features={d}). Using n_components={k} instead."
            )
        self.n_components = k

        if self._center:
            self.mean_ = X.mean(dim=0, keepdim=True)
            Xc = X - self.mean_
        else:
            self.mean_ = None
            Xc = X

        # Principal directions, columns in descending order
        if self.solver == "lowrank":
            _, _, V = torch.pca_lowrank(Xc, q=k, center=False, niter=self.niter)
        else:
            X64 = Xc.double()
            _, evecs = torch.linalg.eigh(X64.T @ X64)  # ascending eigenvalues
            V = evecs.flip(1)[:, :k]

        pivot = V.abs().argmax(dim=0)
        V = V * torch.sign(V[pivot, torch.arange(k, device=V.device)])
        self.components = V.float().contiguous()

        # Explained variance ratio (sklearn's definition for both PCA and SVD)
        Xc64 = Xc.double()
        Z = Xc64 @ self.components.double()
        full_var = Xc64.var(dim=0, unbiased=False).sum()
        ev = Z.var(dim=0, unbiased=False) / full_var
        self.explained_variance_ratio_ = ev.cpu().numpy()

        kind = self._kind + (", L2-normalized input" if self.l2_input else "")
        print(
            f"  {kind} ({self.solver}) fit on {n} samples "
            f"(image={len(train_img)} + text={len(train_txt)}) -> {k} dimensions; "
            f"cumulative explained variance: "
            f"32D={ev[:32].sum().item():.3f}, "
            f"128D={ev[:128].sum().item():.3f}, "
            f"full={ev.sum().item():.3f}"
        )

        self._memo.clear()
        return self

    def _project(self, emb):
        if self.components is None:
            raise RuntimeError(
                f"{type(self).__name__} has not been fitted. "
                "Call .fit() on the training features first."
            )
        if self.mean_ is not None:
            emb = emb - self.mean_
        return emb @ self.components


class PCAReducer(_LinearReducer):
    """(2a) Standard PCA: subtract the training mean, then project.

    Same as sklearn `PCA`. The mean is computed over the stacked images +
    captions ("joint" PCA), so it lies between the two modalities and is
    dominated by captions (5 per image).

    WARNING: joint centering can wreck image->text retrieval. After centering,
    a few captions become "hubs" that are the top match for almost every image,
    while text->image is barely affected. In this project's diagnostic
    (Flickr30k, 1000-image test), i2t R@1 fell from 78.3 to 0.5 on
    unnormalized features (one caption was the top match for 990 of 1000
    images) and to 13.2 with l2_input=True. Use SVDReducer for a projection
    that keeps the original cosine similarities.

    Options: see `_LinearReducer`.
    """

    name = "pca"
    _center = True
    _kind = "PCA"


class SVDReducer(_LinearReducer):
    """(2b) Uncentered truncated SVD: project without subtracting any mean.

    Same as sklearn `TruncatedSVD`. With no centering the projection is a pure
    rotation, so at full dimension the cosine similarities, and therefore the
    retrieval results, are exactly those of the original features (`truncate`
    at full dimension); lower m keeps the highest-energy directions.

    Options: see `_LinearReducer`.
    """

    name = "svd"
    _center = False
    _kind = "uncentered SVD"


class SDMReducer(Reducer):
    """(3) Dimension merging: average clusters of similar dimensions.

    `fit` clusters the 512 feature dimensions hierarchically for every size in
    `dims`; `transform(emb, m)` replaces each cluster by its mean (one output
    dimension per cluster, ordered by first member) and L2-normalizes.

    l2_input : L2-normalize every input row first (see `Reducer`).
    """

    name = "sdm"

    def __init__(self, dims=None, device=None, chunk_size: int = 256, l2_input=False):
        super().__init__(device=device, l2_input=l2_input)
        self.dims = dims
        self.chunk_size = chunk_size
        self.clus = None

    @torch.no_grad()
    def fit(self, train_img, train_txt, data_size=None):
        X = torch.cat([self._input(train_img), self._input(train_txt)], dim=0)
        print(f"All embeddings shape for clustering: {tuple(X.shape)}"
              + (" (L2-normalized input)" if self.l2_input else ""))

        self.clus = calculate_cluster(
            X, num_clusters=self.dims, chunk_size=self.chunk_size
        )
        print(f"Calculated clusters for dimensions: {self.dims}")

        # Cluster dump: clusters/clusters_{data_size}.txt (folder created if needed)
        out = Path("clusters") / f"clusters_{data_size}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"clusters": self.clus}, f, indent=4, default=numpy_converter)
        print(f"Saved clusters to {out}")

        self._memo.clear()
        return self

    @torch.no_grad()
    def transform(self, emb, m):
        if self.clus is None:
            raise RuntimeError(
                "SDMReducer has not been fitted. "
                "Call .fit() on the training features first."
            )
        if self.clus.get(m) is None:
            raise ValueError(
                f"No clustering for m={m}; available: "
                f"{sorted(k for k, v in self.clus.items() if v is not None)}"
            )

        x = self._input(emb)
        ids = torch.as_tensor(self.clus[m], device=self.device)
        means, _ = cluster_means(x, ids)  # [N, m]
        return _to_input_type(l2_normalize(means), emb)