"""Step 0: Environment Self-Check — verify dependencies and GPU availability.
Can be run directly.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main():
    print("Python", sys.version.split()[0])

    ok = True

    # Check required Python packages
    for mod in [
        "torch",
        "open_clip",
        "faiss",
        "numpy",
        "sklearn",
        "yaml",
        "pandas",
        "matplotlib",
        "tqdm",
        "PIL",
        "huggingface_hub",
    ]:
        try:
            m = __import__(mod)
            print(
                "  [ok]      %-13s %s"
                % (mod, getattr(m, "__version__", ""))
            )
        except Exception as e:
            ok = False
            print(
                "  [MISSING] %-13s %s"
                % (mod, e)
            )

    # Check CUDA/GPU
    try:
        import torch

        av = torch.cuda.is_available()

        print(
            "  cuda available:",
            av,
            "| device:",
            torch.cuda.get_device_name(0) if av else "-"
        )

    except Exception:
        pass

    print(
        "OK"
        if ok
        else "Missing dependencies; see above and install them according to the 'Environment' section of the README."
    )


if __name__ == "__main__":
    main()