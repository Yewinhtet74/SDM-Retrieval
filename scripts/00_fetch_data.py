"""Step 0b: Download Flickr30k once to local cfg.paths.data_root
(Karpathy annotation CSV + 31,014 images).

Source: nlphuji/flickr30k:
flickr_annotations_30k.csv (13MB, contains split column) +
flickr30k-images.zip (4.4GB).

Download directly to local_dir instead of the Hub blob cache to avoid
duplicate disk usage caused by Windows without symbolic links.
After completion, {data_root}/_staging can be deleted.
"""

import os
import shutil
import sys
import pathlib
import zipfile

# Must be set before importing huggingface_hub:
# the hf_xet backend buffers the entire 4.4GB zip into memory
# (in practice, the disk file stays at 0 bytes while RSS grows continuously).
# This is unsuitable for a 16GB machine.
# Disable it and use normal HTTP streaming + resume support instead.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import ANNOTATION_CSV, IMAGE_SUBDIR

REPO = "nlphuji/flickr30k"


def main():
    from huggingface_hub import hf_hub_download

    cfg = load_config()
    root = pathlib.Path(cfg.paths.data_root)
    staging, img_dir = root / "_staging", root / IMAGE_SUBDIR

    root.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Downloading annotation CSV...", flush=True)

    csv_src = hf_hub_download(
        REPO,
        ANNOTATION_CSV,
        repo_type="dataset",
        local_dir=str(staging)
    )

    shutil.copy2(csv_src, root / ANNOTATION_CSV)

    print("  ->", root / ANNOTATION_CSV, flush=True)

    print("[2/3] Downloading image zip 4.4GB (resumable)...", flush=True)

    zip_path = hf_hub_download(
        REPO,
        "flickr30k-images.zip",
        repo_type="dataset",
        local_dir=str(staging)
    )

    print(
        f"  -> {zip_path} "
        f"{pathlib.Path(zip_path).stat().st_size / 1e9:.2f} GB",
        flush=True
    )

    print("[3/3] Extracting images into images/ ...", flush=True)

    n = 0

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():

            if info.is_dir():
                continue

            name = pathlib.Path(info.filename).name

            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # The zip was created on macOS and contains a set of
            # __MACOSX/._xxx.jpg resource-fork copies (1:1 with the real images).
            # Flattening by filename would treat them as real images,
            # so they must be skipped.
            if name.startswith("._") or info.filename.startswith("__MACOSX"):
                continue

            target = img_dir / name

            if not (
                target.exists()
                and target.stat().st_size == info.file_size
            ):
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(
                        src,
                        dst,
                        length=1 << 20
                    )

            n += 1

            if n % 5000 == 0:
                print(f"  {n} images", flush=True)

    print(f"Completed: {n} images -> {img_dir}")
    print(f"Temporary staging directory can be deleted: {staging}")


if __name__ == "__main__":
    main()