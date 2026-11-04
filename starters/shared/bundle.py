"""Freeze a bot and its shared runtime into a standalone submission folder."""
from pathlib import Path
import argparse
import shutil

SHARED = Path(__file__).resolve().parent


def bundle_bot(source, destination, checkpoint=None):
    source, destination = Path(source), Path(destination)
    checkpoint = Path(checkpoint) if checkpoint is not None else source / "policy.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No trained weights at {checkpoint}")
    if source.resolve() in (destination.resolve(), *destination.resolve().parents):
        raise ValueError("Bundle destination must be outside the source bot directory")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pt", "*.onnx", "*.json", "shared"))
    shutil.copytree(SHARED, destination / "shared",
                    ignore=shutil.ignore_patterns("__pycache__"))
    # Move tracking imports the canonical referee board and fog move rules.
    shutil.copyfile(SHARED.parent.parent / "referee.py",
                    destination / "shared/referee.py")
    shutil.copyfile(checkpoint, destination / "policy.pt")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bot", type=Path, help="e.g. starters/neural_recurrent")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(bundle_bot(args.bot, args.output))


if __name__ == "__main__":
    main()
