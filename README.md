# SDM-Retrieval

Similarity-Based Dimension Merging for Training-Free Compression of CLIP Embeddings

The code compresses frozen CLIP embeddings to fewer dimensions and measures how much
retrieval quality is lost. Four methods are compared on the same features, the same
training data and the same evaluation code:

| Method | What it does | Fitting |
|---|---|---|
| `baseline` | Native truncation: keep the first *m* dimensions | None |
| `pca` | Standard (centered) joint PCA on stacked image + caption embeddings | Training subset |
| `svd` | Uncentered truncated SVD on stacked image + caption embeddings | Training subset |
| `sdm` | Dimension merging: clusters similar dimensions and averages each cluster | Training subset |

Every method L2-normalizes its output after reducing to *m* dimensions, so retrieval
uses cosine similarity. Quality is measured as R@1/5/10 in both directions on the
standard Flickr30k 1K test set (1,000 images, 5,000 captions). Latency, bytes per vector
and index size are measured with FAISS on the full ~31k-image gallery.

---

## 1. Requirements

- Linux, or WSL 2 on Windows (section 2.4).
- Python 3.12, the version used for all results, installed with conda (section 2.1) or,
  on WSL, with uv (section 2.4).
- A CUDA GPU is recommended for feature extraction (16 GB of GPU memory is enough; see the
  memory notes below). It also runs on CPU, only much more slowly. Evaluation runs fine on
  CPU.
- About 10 GB of free disk space: 4.4 GB for the image zip, about the same for the
  extracted images, plus cached features.

## 2. Setup

All results were produced with the environment below, recorded by
`scripts/00_check_env.py`. These steps reproduce it, and `requirements.txt` pins the same
versions.

| Package | Version |
|---|---|
| Python | 3.12.15 |
| torch / torchvision | 2.6.0+cu124 / 0.21.0 |
| open_clip_torch | 3.3.0 |
| faiss-cpu | 1.15.1 |
| numpy | 2.5.2 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| matplotlib · tqdm · pyyaml · pillow · huggingface_hub | 3.11.2 · 4.70.1 · 6.0.3 · 12.3.0 · 2.2.0 |

torchvision 0.21.0 is the release paired with torch 2.6.0.

Sections 2.1 to 2.3 are the main setup, for a Linux machine or a cloud notebook instance
with conda. Section 2.4 covers WSL on Windows.

### 2.1 Create the environment with conda

```bash
conda create --prefix <env-dir>/sdm python=3.12 -y
conda activate <env-dir>/sdm
which python
```

Replace `<env-dir>` with a folder for environments. On a cloud notebook instance, choose a
folder on the persistent volume, so the environment survives restarts. `which python`
should print `<env-dir>/sdm/bin/python`.

If `conda activate` fails, run `conda init bash`, then open a new terminal (or run
`source ~/.bashrc`) and try again.

### 2.2 Install the packages

With the environment active, go to the repository folder and run:

```bash
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir -r requirements.txt
python scripts/00_check_env.py
```

The check should print the versions in the table above and `cuda available: True`.
`--no-cache-dir` keeps pip from storing a second copy of the large PyTorch download.

- **GPU driver:** the cu124 build needs an NVIDIA driver that supports CUDA 12.4. The
  "CUDA Version" shown by `nvidia-smi` must be 12.4 or higher; newer drivers are fine.
- **`CUDA error: no kernel image is available for execution on the device`:** the GPU is
  newer than this PyTorch build supports (torch 2.6.0+cu124 covers compute capability up
  to 9.0). Keep the environment and replace only PyTorch with a newer CUDA build:
  `pip uninstall -y torch torchvision`, then
  `pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu130`.
  This build needs `nvidia-smi` to show CUDA Version 13.0 or higher. Results computed
  this way use a newer torch than the table above, so record its version.
- **Without a GPU:** install the CPU build instead
  (`pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu`)
  and set `model.device: cpu` and `model.precision: fp32` in `configs/default.yaml`.
  Feature extraction (31k images and 155k captions) then takes hours instead of minutes.
  Everything after extraction runs fine on CPU.
- **`No matching distribution found for torch==2.6.0`:** the active Python is not 3.12.
  Check `python --version`, and make sure the environment is activated.

### 2.3 Register the environment as a Jupyter kernel

```bash
pip install --no-cache-dir ipykernel
python -m ipykernel install --user --name sdm --display-name "Python (sdm)"
```

Then choose the **"Python (sdm)"** kernel in your notebook. If it doesn't appear, reload
the Jupyter page.

### 2.4 Alternative: WSL on Windows

WSL 2 runs a real Linux system, so everything in this README works the same there. Only
the environment and a few details differ.

**Create the environment without conda.** If conda is not installed in WSL, use
[uv](https://docs.astral.sh/uv/) to get Python 3.12. It needs no administrator rights and
leaves the system Python unchanged; recent Ubuntu versions ship Python 3.14, for which
torch 2.6.0 has no build. In the repository folder:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv sdm --python 3.12 --seed
source sdm/bin/activate
python --version
```

This creates the environment in a folder named `sdm` inside the repository. `--seed` puts
`pip` into it, and `python --version` should print `Python 3.12.x`. If an `sdm` folder
from an earlier attempt exists, remove it first (`deactivate`, then `rm -rf sdm`).
Activate the environment again in every new terminal (`source sdm/bin/activate`). Then
continue with sections 2.2 and 2.3.

**Differences on WSL:**

- **Use WSL 2.** In Windows PowerShell, `wsl -l -v` should show `VERSION 2` for your
  distribution.
- **GPU:** install the NVIDIA driver on Windows only. Do not install an NVIDIA driver inside
  WSL; the Windows driver already provides the GPU to WSL. Check with `nvidia-smi` in the
  WSL terminal.
- **Keep the dataset in the Linux file system** (for example `~/datasets/flickr30k`), not on
  a Windows drive under `/mnt/<drive>`. WSL reads Windows drives much more slowly, which
  makes unzipping and feature extraction for 31k images take far longer.
- **Set `paths.data_root` to a Linux path,** such as `/home/<user>/datasets/flickr30k`
  (see section 3).
- **Memory:** WSL gets a share of the computer's RAM, by default about half. If a large
  run (for example SDM with `--train_size 29000`) runs out of memory, raise the limit in
  `.wslconfig` in your Windows user folder.
- **VS Code:** install Microsoft's **WSL** extension, click the `><` icon in the bottom-left
  corner and choose **Connect to WSL**, then **File → Open Folder** and enter the
  repository's WSL path. Install the **Python** and **Jupyter** extensions with
  **Install in WSL**, open a notebook, and choose **Select Kernel → Python Environments →
  sdm**. If `sdm` isn't listed, run **Python: Select Interpreter → Enter interpreter
  path...** and enter `<repository>/sdm/bin/python`.
- **Jupyter Lab:** install it in the environment with `pip install jupyterlab`, start
  `jupyter lab` inside WSL, and open the `http://localhost:8888/...` link it prints in
  your Windows browser.

## 3. Configuration

All settings are in `configs/default.yaml`. The scripts always read this file, so edit it
before running. **Run every script from the repository root**: the config path and the
output paths are relative to it.

| Setting | Meaning |
|---|---|
| `paths.data_root` | Where Flickr30k is downloaded. Set an absolute path for your machine. The default `D:/datasets/flickr30k` is a Windows path; on Linux and WSL it becomes a folder literally named `D:` inside the repository. |
| `paths.output_root` | Where features, subsets and results are written (default `./outputs`) |
| `model.name`, `model.pretrained` | The OpenCLIP encoder (see the table below) |
| `model.device`, `model.precision` | `cuda` / `fp16` by default |
| `data.batch_size_extract` | Batch size for feature extraction. Lower it if you run out of GPU memory. |
| `mrl.dims` | The dimensions *m* to evaluate. **The first value must be the encoder's embedding size**, so that it is the uncompressed reference. |
| `eval.precisions`, `eval.faiss_index` | Precisions (`fp32`, `fp16`, `int8`) and FAISS index type for the latency/storage measurements |

The `adapter`, `loss`, `train`, `lora` and `experiment` sections are left over from the
original project and are not used by these scripts.

### Encoders used

| `model.name` | `model.pretrained` | Embedding size | `mrl.dims` |
|---|---|---|---|
| `ViT-B-32` | `openai` (main encoder) | 512 | `[512, 384, 256, 128, 64]` |
| `RN50` | `openai` | 1024 | `[1024, 768, 512, 384, 256, 128, 64]` |
| `RN50x16` | `openai` | 768 | `[768, 512, 384, 256, 128, 64]` |
| `ViT-L-14-336` | `openai` | 768 | `[768, 512, 384, 256, 128, 64]` |
| `ViT-B-16` | `laion2b_s34b_b88k` | 512 | `[512, 384, 256, 128, 64]` |

Any other OpenCLIP model works too (`open_clip.list_pretrained()` lists them). Add `32`
to `mrl.dims` to also evaluate 32 dimensions.

## 4. Running

### Step 0: download Flickr30k (once)

```bash
python scripts/00_fetch_data.py
```

This downloads the Karpathy-split annotation CSV and the 4.4 GB image zip from the Hugging
Face dataset `nlphuji/flickr30k` into `paths.data_root`, then extracts the 31,014 images.
The download can be resumed. Afterwards, `{data_root}/_staging` can be deleted.

### Step 1 (optional): sanity check

```bash
python scripts/01_sanity_check.py
```

Extracts the test-set features and compares zero-shot R@1/5/10 with the official OpenCLIP
results. The reference numbers are for **`ViT-B-16` / `laion2b_s34b_b88k`**, so the check
only reports PASS with that encoder. With other encoders it reports FAIL even when
everything is correct.

### Step 2: extract features (once per encoder)

```bash
python scripts/02_extract_features.py
```

Encodes the train, val and test images and captions with the configured encoder and caches
them as unnormalized fp32 `.npy` files in `outputs/features/{model}/`, for example
`outputs/features/ViT-B-32_openai/`. Each encoder gets its own folder, with a `model.json`
recording which model produced it. Splits that are already cached are skipped; delete the
folder to re-extract.

### Step 3: evaluate the reducers

```bash
python scripts/03_eval_reducers.py --train_size 1000 --l2
```

| Option | Meaning |
|---|---|
| `--train_size N` | Number of training images used to fit PCA, SVD and SDM (with their 5 captions each). The subset is drawn once per size and seed, saved in `outputs/subsets/`, and reused by every later run. Use `29000` for the full training split. |
| `--methods ...` | Any of `baseline pca svd sdm`, run in the order given (default: all four) |
| `--l2` / `--no-l2` | Also L2-normalize every input embedding before fitting and projecting (off by default). Fitted methods then get `_l2` in their result names. The baseline is unaffected. |
| `--seed S` | Seed for the training subset (default `42`) |
| `--diagnose` | Run only the image-to-text diagnostic (below), then exit |

Examples:

```bash
python scripts/03_eval_reducers.py --train_size 29000 --l2
python scripts/03_eval_reducers.py --train_size 1000 --l2 --methods pca svd
python scripts/03_eval_reducers.py --train_size 10 --l2 --methods sdm
```

The script also accepts `--methods pca_whiten` (PCA with whitening). That method needs a
`src/reducers.py` with whitening support, which the copy in this repository does not have.

SDM is the slowest method to fit. Its cost grows with the number of training rows, so
`--train_size 29000` (174,000 rows) takes much longer than `1000`.

### Diagnostic: why standard PCA hurts image-to-text retrieval

```bash
python scripts/03_eval_reducers.py --train_size 1000 --l2 --diagnose
```

Recomputes R@1 at full dimension for the raw features and for several ways of centering
them (joint mean as in PCA, caption mean only, image mean only, per-modality means, DN),
and reports the modality gap, the offset that joint centering leaves on each modality,
and how many image queries are won by a single "hub" caption.

### Test-set size check

```bash
python scripts/check_test_size.py
```

Measures zero-shot R@1/R@5 of the raw embeddings on random test sets of 1,000 and 6,358
images (the size of a 20% split), to compare with results reported on larger test sets.

### Running from Jupyter

The `evaluate_*.ipynb` notebooks contain the runs for each encoder. In a notebook, use
`%run`, and **do not add a `#` comment after a `%run` command**: Jupyter passes it to the
script as an argument.

```python
%run scripts/02_extract_features.py
%run scripts/03_eval_reducers.py --train_size 1000 --l2 --methods baseline pca svd sdm
```

After replacing any file in `src/`, restart the kernel so the new version is loaded.

## 5. Outputs

| Path | Content |
|---|---|
| `outputs/features/{model}/` | Cached embeddings (`{split}_{image,text}.npy`), ID files, `model.json` |
| `outputs/subsets/train_idx_n{size}_seed{seed}.txt` | The training-image indices used for each subset size |
| `outputs/results/{model}_{method}[_l2]_{size}.csv` | Results of each method, one row per dimension (the baseline is `{model}_baseline_None.csv`) |
| `clusters/clusters_{model}_{size}[_l2].txt` | SDM's cluster assignments (the `clusters/` folder is created in the folder you run from) |

Result columns:

- `model`, `condition`, `dim`
- `i2t_r1`, `i2t_r5`, `i2t_r10`, `t2i_r1`, `t2i_r5`, `t2i_r10`: recall in percent.
  Image-to-text counts a hit if any of the image's 5 captions is retrieved; text-to-image
  if its one image is retrieved.
- `lat_ms_*`, `bytes_*`, `index_mb_*`, `n_gallery`: per-query latency, bytes per vector and
  index size for each precision, on the ~31k-image gallery
- `i2t_r1_ret`, `t2i_r1_ret`: R@1 as a percentage of the R@1 at the largest dimension in
  `mrl.dims`

## 6. Troubleshooting

- **CUDA out of memory during extraction.** Check `nvidia-smi` for other notebooks' kernels
  still holding GPU memory and shut them down. Lower `data.batch_size_extract`
  (64 works for `ViT-L-14-336` on a 16 GB GPU). In a fresh kernel, setting
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing torch reduces
  fragmentation.
- **`Missing feature cache ...`.** Run `scripts/02_extract_features.py` for the configured
  encoder first.
- **`... holds features of X, but the config says Y`.** The feature folder belongs to a
  different model than `configs/default.yaml` names. Fix the config or re-extract.
- **`Model config for '...' not found`.** Use OpenCLIP's model names, e.g. `ViT-B-32`,
  `ViT-L-14-336` (not `ViT-B/32` or `ViT-L/14@336px`). OpenAI never released a ViT-L/32.
- **Results look odd at the first dimension.** Make sure the first value of `mrl.dims`
  equals the encoder's embedding size.

## 7. Project layout

```
configs/default.yaml           settings (edit this)
scripts/
  00_check_env.py              environment check
  00_fetch_data.py             download Flickr30k
  01_sanity_check.py           zero-shot check against official OpenCLIP numbers
  02_extract_features.py       extract and cache CLIP features
  03_eval_reducers.py          evaluate baseline / PCA / SVD / SDM; --diagnose
  check_test_size.py           effect of test-set size on R@1
  03_eval_baseline_pca.py      legacy, see below
src/
  config.py                    load configs/default.yaml
  data.py                      Flickr30k Karpathy split, image/caption loaders
  model.py                     load the frozen OpenCLIP model
  features.py                  extract, cache and load features (one folder per model)
  subset.py                    fixed random training subsets
  reducers.py                  TruncateReducer, PCAReducer, SVDReducer, SDMReducer
  evaluate.py                  evaluate one method at every dimension, save the CSV
  metrics.py                   bidirectional R@K
  retrieval.py                 FAISS latency / storage measurements
  diagnose_i2t.py              image-to-text diagnostic
```

`scripts/03_eval_baseline_pca.py` is an earlier version of step 3, kept for reference.
It passes `center` and `whiten` options that the current `src/reducers.py` no longer has,
so it does not run with this code. Use `03_eval_reducers.py` instead.

## 8. Credits and license

This project builds on [MRL-Retrieval](https://github.com/Azathoth2001/MRL-Retrieval)
by Azathoth2001, released under the MIT License (see `LICENSE`).

CLIP models are loaded with [OpenCLIP](https://github.com/mlfoundations/open_clip).
Flickr30k is downloaded from the Hugging Face dataset `nlphuji/flickr30k`; check the
dataset's terms of use before redistributing it.
