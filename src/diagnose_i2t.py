"""Find out why image->text R@1 is low: centering (hub captions) or the i2t metric.

Recomputes R@1 in both directions from scratch (no metrics.py), for the raw
features and for several kinds of centering, at full dimension. At full
dimension a PCA/SVD projection is only a rotation, so these rows isolate the
effect of centering from the effect of dropping components.

The extra rows test WHY joint centering hurts image->text but not text->image.
Subtracting one shared mean leaves each modality shifted by a different amount:
with 5 captions per image the mean sits close to the captions, so images keep
most of the modality gap as a common offset. If that offset is the cause:
  * "mean of captions only" should hurt i2t (like PCA),
  * "mean of images only" should flip it and hurt t2i instead,
  * "joint mean, equal weights" should split the damage between both,
  * per-modality / DN centering (no leftover offset) should hurt neither.

Run it through the step-3 script (add --l2 to diagnose normalized features):

    python scripts/03_eval_reducers.py --train_size 1000 --diagnose
    python scripts/03_eval_reducers.py --train_size 1000 --l2 --diagnose
"""
import numpy as np


def _l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _scores(img, txt, owner):
    S = _l2(img) @ _l2(txt).T                      # [n_img, n_txt] cosine
    top_cap = S.argmax(1)                          # best caption per image
    i2t = 100 * np.mean(owner[top_cap] == np.arange(len(img)))
    t2i = 100 * np.mean(S.argmax(0) == owner)      # best image per caption
    hits = np.bincount(top_cap, minlength=txt.shape[0])
    return i2t, t2i, hits.max(), np.count_nonzero(hits)


def diagnose(train_img, train_txt, test_img, test_txt, owner):
    tr_img, tr_txt = (np.asarray(a, np.float64) for a in (train_img, train_txt))
    te_img, te_txt = (np.asarray(a, np.float64) for a in (test_img, test_txt))
    owner = np.asarray(owner)

    mu_joint = np.concatenate([tr_img, tr_txt]).mean(0)   # what PCAReducer uses
    mu_img, mu_txt = tr_img.mean(0), tr_txt.mean(0)
    mu_equal = (mu_img + mu_txt) / 2

    variants = {
        "raw (= truncate / SVD at full dim)": (te_img, te_txt),
        "joint centering (= PCAReducer)": (te_img - mu_joint, te_txt - mu_joint),
        "joint mean, equal weights": (te_img - mu_equal, te_txt - mu_equal),
        "mean of captions only": (te_img - mu_txt, te_txt - mu_txt),
        "mean of images only": (te_img - mu_img, te_txt - mu_img),
        "per-modality centering": (te_img - mu_img, te_txt - mu_txt),
        "DN: half per-modality means": (te_img - mu_img / 2, te_txt - mu_txt / 2),
    }

    print(f"modality gap |mean_img - mean_txt| = {np.linalg.norm(mu_img - mu_txt):.3f}")
    print(f"after joint centering, offset left on images = {np.linalg.norm(mu_img - mu_joint):.3f}, "
          f"on captions = {np.linalg.norm(mu_txt - mu_joint):.3f}")
    print(f"{len(te_img)} test images, {len(te_txt)} captions\n")
    print(f"{'variant':38s} {'i2t R@1':>8s} {'t2i R@1':>8s} "
          f"{'top hub':>8s} {'distinct':>9s}   caption-norm CV")
    for name, (I, T) in variants.items():
        i2t, t2i, hub, distinct = _scores(I, T, owner)
        n = np.linalg.norm(T, axis=1)
        print(f"{name:38s} {i2t:8.2f} {t2i:8.2f} {hub:8d} {distinct:9d}   {n.std() / n.mean():.3f}")
    print("\ntop hub  = most image queries won by a single caption (healthy: small)")
    print("distinct = captions that are top-1 for at least one image (healthy: close to #images)")