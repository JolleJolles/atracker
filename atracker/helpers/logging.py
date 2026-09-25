"""Consistent local timestamps for ATracker console messages."""

import time
import sys


def lineprint(text, stamp=True, newline=True, date=True, **kwargs):
    """Print a message with date/time and no machine-name prefix."""
    kwargs.pop("label", None)
    if stamp:
        fmt = "%Y-%m-%d %H:%M:%S" if date else "%H:%M:%S"
        text = f"{time.strftime(fmt)} - {text}"
    kwargs.setdefault("end", "\n" if newline else " ")
    print(text, **kwargs)


class TrackingProgress:
    """Report frame milestones; replace terminal lines only for a single worker."""

    def __init__(self, prefix, pooled=False, interval=1000):
        self.prefix = prefix
        self.interval = interval
        self.last_processed = 0
        self.started = time.monotonic()
        self.inline = not pooled and sys.stdout.isatty()
        self.width = 0

    def update(self, frame, stop, processed):
        if processed - self.last_processed < self.interval:
            return
        self.last_processed = processed
        now = time.monotonic()
        fps = processed / max(now - self.started, 1e-9)
        message = (f"{self.prefix}Frame {frame}/{int(stop)} — {fps:.1f} FPS")
        if self.inline:
            self.width = max(self.width, len(message))
            print("\r" + message.ljust(self.width), end="", flush=True)
        else:
            lineprint(message, date=False, flush=True)

    def clear(self):
        if self.width:
            print("\r" + " " * self.width + "\r", end="", flush=True)
            self.width = 0
