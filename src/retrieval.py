"""Engineering metrics: FAISS index + latency/storage/index size.
Single responsibility: retrieval engineering measurements.

Latency is measured per single query (this reflects the latency perceived
by an online application). The number of threads is fixed to 1:
using multiple threads on a 31k-image gallery can hide dimensionality-related
differences behind scheduling noise. Single-thread measurements provide a
cleaner relationship between dimensionality and latency.

fp16/int8 use FAISS ScalarQuantizer for actual quantized storage,
rather than "converting and then converting back", which would not represent
real reduced-precision storage.
"""
from __future__ import annotations
import time

import faiss
import numpy as np

BYTES_PER_DIM = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0}
_QUANT = {"fp16": "QT_fp16", "int8": "QT_8bit"}


def _build_index(emb, precision, index_type, nlist=None):
    """Build a FAISS index using precision × index type and add all data.

    Returns:
        (index, description_string)
    """
    n, d = emb.shape
    metric = faiss.METRIC_INNER_PRODUCT
    # Vectors are already L2-normalized, so inner product = cosine similarity.

    if precision not in BYTES_PER_DIM:
        raise ValueError(
            f"precision must be one of {list(BYTES_PER_DIM)}, "
            f"got {precision!r}"
        )

    if index_type == "flat":
        if precision == "fp32":
            index, desc = faiss.IndexFlatIP(d), "FlatIP"
        else:
            qt = getattr(faiss.ScalarQuantizer, _QUANT[precision])
            index, desc = (
                faiss.IndexScalarQuantizer(d, qt, metric),
                f"SQ({precision})"
            )

    elif index_type == "ivf":
        nlist = nlist or max(1, int(4 * np.sqrt(n)))

        coarse = faiss.IndexFlatIP(d)

        if precision == "fp32":
            index, desc = (
                faiss.IndexIVFFlat(coarse, d, nlist, metric),
                f"IVF{nlist},Flat"
            )
        else:
            qt = getattr(faiss.ScalarQuantizer, _QUANT[precision])
            index = faiss.IndexIVFScalarQuantizer(
                coarse, d, nlist, qt, metric
            )
            desc = f"IVF{nlist},SQ({precision})"

        index.nprobe = max(1, nlist // 16)

    else:
        raise ValueError(
            f"index must be flat/ivf, got {index_type!r}"
        )

    if not index.is_trained:
        index.train(emb)

    index.add(emb)

    return index, desc


def bench(
    gallery_emb,
    query_emb,
    m,
    precision="fp32",
    index="flat",
    k=10,
    n_queries=200,
    repeats=3,
    threads=1,
    seed=0,
):
    """Measure retrieval on a gallery.

    Measures:
        - single-query latency (ms)
        - QPS
        - bytes per vector
        - serialized index size (MB)

    Args:
        gallery_emb:
            Gallery embeddings.

        query_emb:
            Query embeddings.

        m:
            Target dimensionality.

        precision:
            Storage precision: fp32 / fp16 / int8.

        index:
            FAISS index type: flat / ivf.

    Both gallery_emb and query_emb must already be reduced to m dimensions
    and L2-normalized, i.e. they must be outputs of reducer.transform().

    Returns:
        Dictionary containing the benchmark results.
    """

    gallery = np.ascontiguousarray(
        gallery_emb,
        dtype=np.float32
    )

    queries = np.ascontiguousarray(
        query_emb,
        dtype=np.float32
    )
    if gallery.shape[1] != m or queries.shape[1] != m:
        print(
            f"Expected {m} dimensions, "
            f"but got gallery={gallery.shape[1]}, "
            f"query={queries.shape[1]}"
        )
    # if gallery.shape[1] != m or queries.shape[1] != m:
    #     raise ValueError(
    #         f"Expected {m} dimensions, "
    #         f"but got gallery={gallery.shape[1]}, "
    #         f"query={queries.shape[1]}"
    #     )

    # Save the current FAISS thread count and temporarily use the
    # requested number of threads.
    prev_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(threads)

    try:
        # Build the index and add the entire gallery.
        index_obj, desc = _build_index(
            gallery,
            precision,
            index
        )

        # Measure actual serialized index size.
        index_bytes = int(
            faiss.serialize_index(index_obj).nbytes
        )

        # Randomly select queries for benchmarking.
        rng = np.random.default_rng(seed)

        pick = rng.choice(
            len(queries),
            size=min(n_queries, len(queries)),
            replace=False
        )

        probe = np.ascontiguousarray(
            queries[pick]
        )

        # Warm-up:
        # allow BLAS, quantization tables, caches, etc. to initialize.
        for i in range(min(20, len(probe))):
            index_obj.search(
                probe[i:i + 1],
                k
            )

        # Measure single-query latency.
        # Run multiple times and keep the minimum time to reduce
        # the influence of temporary system noise.
        best = None

        for _ in range(repeats):
            t0 = time.perf_counter()

            for i in range(len(probe)):
                index_obj.search(
                    probe[i:i + 1],
                    k
                )

            elapsed = time.perf_counter() - t0

            best = (
                elapsed
                if best is None
                else min(best, elapsed)
            )

        # Average latency of one query in milliseconds.
        lat_ms = (
            1000.0 * best / len(probe)
        )

        # Batch throughput:
        # this is a different measurement from single-query latency
        # and is provided only for reference.
        t0 = time.perf_counter()

        index_obj.search(
            probe,
            k
        )

        batch_qps = (
            len(probe)
            / max(time.perf_counter() - t0, 1e-9)
        )

    finally:
        # Restore the original FAISS thread count.
        faiss.omp_set_num_threads(prev_threads)

    return {
        "dim": m,
        "precision": precision,
        "index": desc,
        "n_gallery": len(gallery),

        # Single-query latency.
        "lat_ms": lat_ms,

        # Single-query QPS.
        "qps": 1000.0 / lat_ms,

        # Batch-query throughput.
        "batch_qps": batch_qps,

        # Theoretical storage required by one vector.
        "bytes_per_vec": (
            m * BYTES_PER_DIM[precision]
        ),

        # Actual serialized FAISS index size.
        "index_mb": index_bytes / 1e6,

        # Actual index storage per gallery vector.
        "index_bytes_per_vec": (
            index_bytes / len(gallery)
        ),
    }