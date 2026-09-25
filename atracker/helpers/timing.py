"""Optional, bounded-memory wall-clock timings for the tracking loop."""

from collections import defaultdict
from time import perf_counter


class TrackingTiming:
    def __init__(self, interval=0, prefix="", emit=print):
        self.interval = interval
        self.prefix = prefix
        self.emit = emit
        self.seconds = defaultdict(float)
        self.frames = 0
        self.last = None

    def start(self):
        if self.interval:
            self.last = perf_counter()

    def mark(self, stage):
        if self.interval:
            now = perf_counter()
            self.seconds[stage] += now - self.last
            self.last = now

    def frame(self, frame_nr):
        if self.interval:
            self.frames += 1
            if self.frames >= self.interval:
                self.report(frame_nr)

    def report(self, frame_nr):
        if not self.interval or not self.frames:
            return
        total = sum(self.seconds.values())
        parts = [f"{name}={seconds * 1000 / self.frames:.1f}ms ({seconds / total:.0%})"
                 for name, seconds in sorted(self.seconds.items(), key=lambda item: -item[1])]
        fps = self.frames / total if total else 0
        self.emit(f"{self.prefix}timing through frame {frame_nr}: "
                  f"{self.frames} processed, {fps:.1f}fps; " + ", ".join(parts))
        self.seconds.clear()
        self.frames = 0
