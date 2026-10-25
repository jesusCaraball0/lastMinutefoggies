#!/usr/bin/env python3
"""Replay the full boards in a referee JSON game in the terminal."""

import argparse
import json
import math
from pathlib import Path
import sys
import time


def render_frame(game, frame, final=False):
    board = frame["board"]
    lines = [
        f"White: {game.get('white', 'White')} | Black: {game.get('black', 'Black')}",
        f"Ply {frame['ply']} | Move: {frame.get('move') or 'starting position'}",
        "",
    ]
    for rank in range(7, -1, -1):
        row = board[rank * 8:(rank + 1) * 8].replace("-", ".")
        lines.append(f"{rank + 1}  " + " ".join(row))
    lines.extend(["", "   a b c d e f g h", "", "Uppercase: White | lowercase: Black"])
    if final:
        lines.extend(["", f"Result: {game.get('result', '?')} — {game.get('reason', 'Game ended')}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game", type=Path, help="JSON replay written by referee.py")
    parser.add_argument("--delay", type=float, default=1.25,
                        help="seconds between frames (default: 2)")
    args = parser.parse_args()
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay must be a finite, nonnegative number")
    try:
        game = json.loads(args.game.read_text())
    except (OSError, ValueError) as error:
        parser.error(str(error))
    frames = game.get("frames") if isinstance(game, dict) else None
    if not isinstance(frames, list) or not frames:
        parser.error("replay must contain a nonempty frames list")
    for index, frame in enumerate(frames):
        if (not isinstance(frame, dict) or "ply" not in frame
                or not isinstance(frame.get("board"), str)
                or len(frame["board"]) != 64
                or not set(frame["board"]) <= set("-PNBRQKpnbrqk")):
            parser.error(f"invalid board or missing ply in frame {index}")
    try:
        for index, frame in enumerate(frames):
            final = index == len(frames) - 1
            # Clear the terminal and move the cursor to the top-left corner.
            clear = "\033[2J\033[H" if sys.stdout.isatty() else ""
            print(clear + render_frame(game, frame, final), flush=True)
            if not final:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
