"""Commit completed CSV output without exposing a partially written final file."""
import os
import tempfile


def write_csv_atomic(data, path, **kwargs):
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix="." + os.path.basename(path) + "-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        data.to_csv(temporary, **kwargs)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
