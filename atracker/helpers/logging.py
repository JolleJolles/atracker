"""Consistent local timestamps for ATracker console messages."""

import time


def lineprint(text, stamp=True, newline=True, date=True, **kwargs):
    """Print a message with date/time and no machine-name prefix."""
    kwargs.pop("label", None)
    if stamp:
        fmt = "%Y-%m-%d %H:%M:%S" if date else "%H:%M:%S"
        text = f"{time.strftime(fmt)} - {text}"
    kwargs.setdefault("end", "\n" if newline else " ")
    print(text, **kwargs)
