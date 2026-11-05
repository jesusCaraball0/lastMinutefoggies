"""Adapt Python agents and existing wire-protocol bots to referee.run_match.

No board, move, visibility, clock or terminal rules are implemented here.
"""
from dataclasses import dataclass
import re
import time
from typing import Callable, Protocol

import referee


class Agent(Protocol):
    def reset(self, side: bool, base: float, increment: float) -> None: ...
    def observe(self, view: str, ply: int) -> None: ...
    def act(self, remaining: float) -> str: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class Command:
    """An existing executable implementing fow / ready / observation / move."""
    argv: tuple


class InProcessBot:
    """The minimal Bot interface consumed by the provided referee."""
    def __init__(self, factory, side, base, increment, ready, startup_timeout,
                 initialization_timeout, timeline, turn_control_factory):
        if ready is not None or turn_control_factory is not None:
            raise ValueError('In-process agents do not use container adapters')
        self.failure = None
        self.stderr_capture = referee.StderrCapture()
        self.stderr_excerpt = self.stderr_capture.excerpt
        self.agent = factory()
        started = time.monotonic()
        try:
            self.agent.reset(side, base, increment)
            if time.monotonic() - started > initialization_timeout:
                raise referee.BotFailure('Bot startup timed out')
        except BaseException:
            self.agent.close()
            raise

    def observe(self, view, ply):
        self.agent.observe(view, ply)

    def move(self, remaining):
        started = time.monotonic()
        move = self.agent.act(remaining)
        elapsed = time.monotonic() - started
        # Wire grammar only. The referee itself validates movement and clocks.
        if not isinstance(move, str) or re.fullmatch(r'[a-h][1-8][a-h][1-8][qrbn]?', move) is None:
            raise referee.BotFailure('Malformed move response')
        return move, elapsed

    def close(self):
        self.agent.close()

    def protocol_snapshot(self):
        return {'transcript': [], 'truncated': False}


def bot_factory(spec, *args):
    if isinstance(spec, Command):
        return referee.Bot(list(spec.argv), *args)
    return InProcessBot(spec, *args)


def play_game(white: Callable[[], Agent], black: Callable[[], Agent], *,
              max_plies=256, base=180., increment=2., **kwargs):
    """Execute the provided game unchanged, returning its standard replay.

    max_plies is the referee's explicit game cap (a draw), not a bootstrapped
    rollout cutoff. Complete episodes are used by the initial PPO collector.
    """
    return referee.run_match(white, black, max_plies=max_plies, base=base,
                             increment=increment, bot_factory=bot_factory, **kwargs)
