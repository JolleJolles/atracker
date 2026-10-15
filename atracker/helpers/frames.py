"""Frame labels in saved data are one-based; OpenCV/editor indices are zero-based."""


def frame_to_index(frame):
    """Convert a one-based frame label to a zero-based seek index."""
    return int(frame) - 1


def index_to_frame(index):
    """Convert a zero-based seek index to a one-based frame label."""
    return int(index) + 1
