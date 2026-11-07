"""History made only from the player's observed views, newest frame first."""
from collections import deque


class ObservationHistory:
    def __init__(self, frames=1):
        if not isinstance(frames, int) or frames < 1:
            raise ValueError('History frame count must be a positive integer')
        self.frames = frames
        self.views = deque(maxlen=frames)

    def observe(self, view):
        if len(view) != 64 or not set(view) <= set('PNBRQKpnbrqk?-'):
            raise ValueError('Expected a 64-square fog observation')
        self.views.appendleft(view)

    def encoded_view(self):
        # Missing history stays unknown; never copy a future or another side's view.
        return ''.join(list(self.views) + ['?' * 64] * (self.frames-len(self.views)))
