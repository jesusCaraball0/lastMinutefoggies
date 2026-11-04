#!/usr/bin/env python3
"""Play two bot directories against each other, alternating colors."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time

from referee import run_match
from starters.shared.bundle import bundle_bot

ROOT = Path(__file__).resolve().parent


def bot_directory(value):
    path = Path(value)
    if not path.is_dir():
        path = ROOT / "starters" / value
    if not path.is_dir() or not (path / "run.sh").is_file():
        raise argparse.ArgumentTypeError(f"{value}: expected a bot directory containing run.sh")
    return path.resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bot_a", type=bot_directory, help="bot directory or starter name, e.g. neural")
    parser.add_argument("bot_b", type=bot_directory, help="bot directory or starter name, e.g. neural_recurrent")
    parser.add_argument("--games", type=int, default=20, help="positive even game count (default: 20)")
    parser.add_argument("--base-time", type=float, default=180)
    parser.add_argument("--increment", type=float, default=2)
    parser.add_argument("--max-plies", type=int, default=1000)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "evaluation_results")
    args = parser.parse_args(argv)
    if args.games < 2 or args.games % 2:
        parser.error("--games must be a positive even number to balance colors")
    if not math.isfinite(args.base_time) or args.base_time <= 0:
        parser.error("--base-time must be finite and positive")
    if not math.isfinite(args.increment) or args.increment < 0:
        parser.error("--increment must be finite and nonnegative")
    if args.max_plies < 1:
        parser.error("--max-plies must be positive")
    return args


def snapshot_bot(source, destination):
    checkpoint = source / "policy.pt"
    if checkpoint.is_file():
        bundle_bot(source, destination)
    else:
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__"))
    frozen = destination / "policy.pt"
    return {
        "directory": str(source),
        "name": source.name,
        "sha256": hashlib.sha256(frozen.read_bytes()).hexdigest() if frozen.is_file() else None,
    }


def main(argv=None):
    args = parse_args(argv)
    totals = Counter(wins=0, losses=0, draws=0, bot_a_failures=0, bot_b_failures=0)
    colors = {color: Counter(wins=0, losses=0, draws=0) for color in ("white", "black")}
    reasons = Counter()
    lengths = []
    started = time.monotonic()
    with TemporaryDirectory(prefix="fow-evaluation-") as directory:
        bundles = [Path(directory) / side for side in ("bot_a", "bot_b")]
        # Freeze both bots before starting the series; ongoing training cannot
        # change either checkpoint halfway through evaluation.
        bots = [snapshot_bot(source, destination) for source, destination in
                zip((args.bot_a, args.bot_b), bundles)]
        names = [f"{bot['name']} ({side})" for bot, side in zip(bots, ("A", "B"))]
        commands = [["bash", str(bundle / "run.sh")] for bundle in bundles]
        # run.sh uses python3; keep it in the evaluator's Python environment.
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
        output = args.results_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output.mkdir(parents=True)
        for name, bot in zip(names, bots):
            print(f"{name}: {bot['directory']}\nCheckpoint SHA256: {bot['sha256'] or 'no policy.pt'}", flush=True)
        print(f"Playing {args.games} games ({args.games // 2} per color), "
              f"{args.base_time:g}+{args.increment:g}. Replays: {output}", flush=True)
        print(f"Results below are from {names[0]}'s perspective.", flush=True)
        for game in range(args.games):
            color = "white" if game % 2 == 0 else "black"
            white, black = (0, 1) if color == "white" else (1, 0)
            game_started = time.monotonic()
            replay = run_match(
                commands[white], commands[black], base=args.base_time, increment=args.increment,
                white_name=names[white], black_name=names[black], max_plies=args.max_plies,
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
                totals["bot_a_failures" if failed == color else "bot_b_failures"] += 1
            label = {"wins": "WIN", "losses": "LOSS", "draws": "DRAW"}[outcome]
            print(
                f"Game {game + 1:2d}/{args.games} | A {color:5s} | {label:4s} | "
                f"{plies:4d} plies | {time.monotonic() - game_started:.1f}s | {replay['reason']} | "
                f"W/L/D {totals['wins']}/{totals['losses']}/{totals['draws']}", flush=True,
            )
            if failed:
                diagnostics = replay.get("privateDiagnostics", {}).get(failed, "")
                if diagnostics:
                    print(f"{failed.capitalize()} bot diagnostics:\n{diagnostics}", flush=True)

    score = (totals["wins"] + totals["draws"] / 2) / args.games
    elapsed = time.monotonic() - started
    print(f"\n{names[0]} vs {names[1]}: {totals['wins']} wins, {totals['losses']} losses, {totals['draws']} draws")
    print(f"Win rate: {totals['wins'] / args.games:.1%} | Score (win=1, draw=0.5): {score:.1%}")
    for color, counts in colors.items():
        print(f"As {color}: {counts['wins']} wins, {counts['losses']} losses, {counts['draws']} draws")
    print(f"Plies: average {sum(lengths) / args.games:.1f}, min {min(lengths)}, max {max(lengths)}")
    print(f"Bot failures: A {totals['bot_a_failures']}, B {totals['bot_b_failures']}")
    print(f"Wall time: {elapsed:.1f}s total, {elapsed / args.games:.1f}s/game")
    print("Termination reasons:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")
    (output / "summary.json").write_text(json.dumps({
        "bot_a": bots[0], "bot_b": bots[1], "perspective": "bot_a", "games": args.games,
        "time_control": {"base": args.base_time, "increment": args.increment},
        "max_plies": args.max_plies, "totals": dict(totals),
        "by_color": {color: dict(counts) for color, counts in colors.items()},
        "reasons": dict(reasons), "plies": lengths, "score": score, "wall_seconds": elapsed,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
