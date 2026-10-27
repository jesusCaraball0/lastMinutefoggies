#!/usr/bin/env python3
"""Play the neural deliverable against baseline.py for 20 games."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time

from referee import run_match


ROOT = Path(__file__).resolve().parent
NEURAL_DIR = ROOT / "starters/neural"
MODEL_PATH = NEURAL_DIR / "policy.pt"
GAMES = 20
BASE_TIME = 180
INCREMENT = 2
MAX_PLIES = 1000
SEED = 7980
RESULTS_DIR = ROOT / "evaluation_results"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"No trained weights at {MODEL_PATH}. Run training/train.py first.")
    if GAMES < 2 or GAMES % 2:
        raise ValueError("GAMES must be a positive even number to balance colors")
    output = RESULTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    totals = Counter()
    colors = {"white": Counter(), "black": Counter()}
    reasons = Counter()
    lengths = []
    started = time.monotonic()
    with TemporaryDirectory(prefix="fow-evaluation-") as directory:
        # Evaluate an independent copy of the complete deliverable. Atomic
        # training checkpoint updates cannot change the weights mid-series.
        bundle = Path(directory) / "neural"
        shutil.copytree(NEURAL_DIR, bundle,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pt", "*.onnx", "*.json"))
        shutil.copyfile(MODEL_PATH, bundle / "policy.pt")
        digest = hashlib.sha256((bundle / "policy.pt").read_bytes()).hexdigest()
        neural = [sys.executable, "-B", "-u", str(bundle / "bot.py")]
        print(f"Model: {MODEL_PATH}\nSHA256: {digest}", flush=True)
        print(f"Playing {GAMES} games ({GAMES // 2} per color), {BASE_TIME}+{INCREMENT}. "
              f"Replays: {output}", flush=True)
        for game in range(GAMES):
            color = "white" if game % 2 == 0 else "black"
            # Each pair shares a baseline RNG seed and swaps colors.
            baseline = [sys.executable, "-B", "-u", str(ROOT / "baseline.py"),
                        "--seed", str(SEED + game // 2)]
            white, black = (neural, baseline) if color == "white" else (baseline, neural)
            game_started = time.monotonic()
            replay = run_match(
                white, black, base=BASE_TIME, increment=INCREMENT,
                white_name="Neural" if color == "white" else "Baseline",
                black_name="Neural" if color == "black" else "Baseline", max_plies=MAX_PLIES,
            )
            (output / f"game-{game + 1:02d}.json").write_text(json.dumps(replay) + "\n")
            result = replay["result"]
            if result not in ("1-0", "0-1", "1/2-1/2"):
                raise RuntimeError(f"Game {game + 1} has no result: {replay['reason']}")
            winning_result = "1-0" if color == "white" else "0-1"
            outcome = "draws" if result == "1/2-1/2" else "wins" if result == winning_result else "losses"
            totals[outcome] += 1
            colors[color][outcome] += 1
            reasons[replay["reason"]] += 1
            plies = len(replay["frames"]) - 1
            lengths.append(plies)
            failed = replay.get("failedSide")
            if failed:
                totals["neural_failures" if failed == color else "baseline_failures"] += 1
            label = {"wins": "WIN", "losses": "LOSS", "draws": "DRAW"}[outcome]
            print(
                f"Game {game + 1:2d}/{GAMES} | neural {color:5s} | {label:4s} | "
                f"{plies:4d} plies | {time.monotonic() - game_started:.1f}s | {replay['reason']} | "
                f"W/L/D {totals['wins']}/{totals['losses']}/{totals['draws']}", flush=True,
            )
            if failed:
                diagnostics = replay.get("privateDiagnostics", {}).get(failed, "")
                if diagnostics:
                    print(f"{failed.capitalize()} bot diagnostics:\n{diagnostics}", flush=True)

    score = (totals["wins"] + totals["draws"] / 2) / GAMES
    elapsed = time.monotonic() - started
    print(f"\nNeural vs baseline: {totals['wins']} wins, {totals['losses']} losses, {totals['draws']} draws")
    print(f"Win rate: {totals['wins'] / GAMES:.1%} | Score (win=1, draw=0.5): {score:.1%}")
    for color, counts in colors.items():
        print(f"As {color}: {counts['wins']} wins, {counts['losses']} losses, {counts['draws']} draws")
    print(f"Plies: average {sum(lengths) / GAMES:.1f}, min {min(lengths)}, max {max(lengths)}")
    print(f"Bot failures: neural {totals['neural_failures']}, baseline {totals['baseline_failures']}")
    print(f"Wall time: {elapsed:.1f}s total, {elapsed / GAMES:.1f}s/game")
    print("Termination reasons:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")
    (output / "summary.json").write_text(json.dumps({
        "model": str(MODEL_PATH), "sha256": digest, "games": GAMES,
        "time_control": {"base": BASE_TIME, "increment": INCREMENT},
        "max_plies": MAX_PLIES, "seed": SEED, "totals": dict(totals),
        "by_color": {color: dict(counts) for color, counts in colors.items()},
        "reasons": dict(reasons), "plies": lengths, "score": score, "wall_seconds": elapsed,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
