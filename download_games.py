#!/usr/bin/env python3
"""Download completed FOW replays where both teams' current Elo is >= 1800.

Install requirements.txt first. Run: python download_games.py
Ratings are a single leaderboard snapshot, not the ratings at game time.
"""

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


from tqdm import tqdm


BASE_URL = "https://6s890.lids.mit.edu"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "game_data"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    """Replace complete files atomically so interrupted downloads are resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def retry_after_seconds(value):
    if not value:
        return 0.0
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return max(0.0, seconds) if math.isfinite(seconds) else 0.0


class Client:
    def __init__(self, interval=1 / 9, retries=5, listing_interval=1.0):
        self.interval = interval
        self.listing_interval = listing_interval
        self.retries = retries
        self.next_request = 0.0
        self.next_listing = 0.0

    def get(self, path, parameters=None):
        url = BASE_URL + path
        if parameters:
            url += "?" + urlencode(parameters)
        is_listing = path == "/api/games"
        for attempt in range(self.retries + 1):
            deadline = max(self.next_request, self.next_listing if is_listing else 0.0)
            time.sleep(max(0.0, deadline - time.monotonic()))
            # All requests share the public-read budget, including retries.
            started = time.monotonic()
            self.next_request = started + self.interval
            if is_listing:
                self.next_listing = started + self.listing_interval
            request = Request(url, headers={"Accept": "application/json",
                                            "User-Agent": "fow-replay-downloader/1.0"})
            retry_after = 0.0
            try:
                with urlopen(request, timeout=30) as response:
                    return json.load(response)
            except HTTPError as error:
                status = error.code
                retry_after = retry_after_seconds(error.headers.get("Retry-After"))
                error.close()
                if status not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise
                detail = f"HTTP {status}"
            except (URLError, TimeoutError, OSError) as error:
                if attempt == self.retries:
                    raise
                detail = str(error)
            delay = max(retry_after, 2 ** attempt + random.uniform(0, 0.5))
            tqdm.write(f"{detail}; retry {attempt + 1}/{self.retries} in {delay:.1f}s",
                       file=sys.stderr)
            self.next_request = max(self.next_request, time.monotonic() + delay)


def valid_replay(replay, game_id):
    if not isinstance(replay, dict) or replay.get("id") != game_id:
        return False
    frames = replay.get("frames")
    return (isinstance(frames, list) and bool(frames)
            and all(isinstance(frame, dict)
                    and isinstance(frame.get("board"), str)
                    and len(frame["board"]) == 64
                    and all(key in frame for key in ("ply", "move", "white", "black", "clocks"))
                    for frame in frames))


def download(output, minimum_elo, interval, listing_interval=1.0):
    client = Client(interval, listing_interval=listing_interval)
    leaderboard = client.get("/api/leaderboard")
    snapshot_at = utc_now()
    # IDs stay stable when names and active submission versions change.
    players = {player["id"]: player for player in leaderboard["players"]}
    ratings = {team_id: player.get("rating") for team_id, player in players.items()}
    eligible = {team_id for team_id, rating in ratings.items()
                if isinstance(rating, (int, float)) and math.isfinite(rating)
                and rating >= minimum_elo}
    metadata = output / "_metadata"
    save_json(metadata / "leaderboard.json", {"fetchedAt": snapshot_at,
                                             "leaderboard": leaderboard})
    save_json(metadata / "player_elos.json", {"fetchedAt": snapshot_at,
                                             "minimumElo": minimum_elo,
                                             "ratings": ratings})
    stats = dict(scanned=0, qualifying=0, downloaded=0, alreadySaved=0,
                 unknownTeams=0, unavailable=0, pages=0)
    unavailable_ids = []
    cursor = None
    seen_cursors = set()
    seen_games = set()
    print(f"{len(eligible)}/{len(players)} teams have current Elo >= {minimum_elo:g}.",
          flush=True)
    with tqdm(desc="Games scanned", unit="game", dynamic_ncols=True) as progress:
        while True:
            parameters = dict(sort="played", order="desc", limit=100)
            if cursor is not None:
                parameters["cursor"] = cursor
            page = client.get("/api/games", parameters)
            stats["pages"] += 1
            if stats["pages"] == 1 and page.get("summary") is not None:
                progress.total = page["summary"].get("total")
                progress.refresh()
            for game in page["games"]:
                game_id = game["id"]
                if not isinstance(game_id, str) or not re.fullmatch(r"[a-z0-9]{8}", game_id):
                    raise ValueError(f"Invalid game ID: {game_id!r}")
                if game_id in seen_games:
                    continue
                seen_games.add(game_id)
                stats["scanned"] += 1
                progress.update(1)
                progress.set_postfix(qualifying=stats["qualifying"],
                                     downloaded=stats["downloaded"],
                                     saved=stats["alreadySaved"], refresh=False)
                white, black = game.get("whitePlayerId"), game.get("blackPlayerId")
                if white not in players or black not in players:
                    # Never guess identity from historical display names.
                    stats["unknownTeams"] += 1
                    continue
                if white not in eligible or black not in eligible:
                    continue
                stats["qualifying"] += 1
                path = output / f"{game_id}.json"
                if path.exists():
                    try:
                        existing = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        existing = None
                    if valid_replay(existing, game_id):
                        stats["alreadySaved"] += 1
                        continue
                try:
                    replay = client.get(f"/api/games/{game_id}")
                except HTTPError as error:
                    if error.code != 404:
                        raise
                    stats["unavailable"] += 1
                    unavailable_ids.append(game_id)
                    tqdm.write(f"Replay {game_id} is unavailable (404); skipping.", file=sys.stderr)
                    continue
                if not valid_replay(replay, game_id):
                    raise ValueError(f"Replay {game_id} is missing required frame data")
                # Keep all API fields intact, adding provenance for this selection.
                replay["downloadMetadata"] = {
                    "downloadedAt": utc_now(), "leaderboardFetchedAt": snapshot_at,
                    "minimumElo": minimum_elo,
                    "whiteCurrentElo": ratings[white], "blackCurrentElo": ratings[black],
                    "gameListing": game,
                }
                save_json(path, replay)
                stats["downloaded"] += 1
            progress.set_postfix(qualifying=stats["qualifying"],
                                 downloaded=stats["downloaded"],
                                 saved=stats["alreadySaved"], unavailable=stats["unavailable"])
            cursor = page["nextCursor"]
            if cursor is None:
                break
            if cursor in seen_cursors:
                raise ValueError("API repeated a pagination cursor; stopping")
            seen_cursors.add(cursor)
    save_json(metadata / "last_run.json", {
        "finishedAt": utc_now(), "leaderboardFetchedAt": snapshot_at,
        "minimumElo": minimum_elo, "stats": stats, "unavailableGameIds": unavailable_ids,
    })
    print(f"Finished. Replays: {output.resolve()}\n{json.dumps(stats)}", flush=True)
    return 1 if stats["unavailable"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="destination directory (default: game_data beside this script)")
    parser.add_argument("--min-elo", type=float, default=1800,
                        help="inclusive current Elo threshold (default: 1800)")
    parser.add_argument("--interval", type=float, default=1 / 9,
                        help="minimum seconds between all requests; at least 1/9 (default: 1/9)")
    parser.add_argument("--listing-interval", type=float, default=1.0,
                        help="minimum seconds between game-listing requests; at least 1 (default: 1)")
    args = parser.parse_args()
    if not math.isfinite(args.min_elo):
        parser.error("--min-elo must be finite")
    if not math.isfinite(args.interval) or args.interval < 1 / 9:
        parser.error("--interval must be finite and at least 1/9 second")
    if not math.isfinite(args.listing_interval) or args.listing_interval < 1:
        parser.error("--listing-interval must be finite and at least 1 second")
    try:
        return download(args.output, args.min_elo, args.interval, args.listing_interval)
    except KeyboardInterrupt:
        print("\nStopped. Saved replays are preserved; rerun to skip them.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Download stopped: {error}. Saved replays are preserved.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
