"""Discover incoming recordings using an existing project row as a setup template."""
from ast import literal_eval
from pathlib import Path
import os
import shutil
import time

import cv2
import pandas as pd

from .logging import lineprint
from .media import bg_extract


def watch_folder(project, folder, template_ind, poll_interval=30, settle_seconds=60,
                 extract_background=True):
    """Watch completed MP4s until Ctrl+C; never modify incoming source files.

    A file must have unchanged size/mtime in two scans and be older than
    settle_seconds. Copy/rename completed recordings into this folder when the
    recorder may pause writes for longer than that interval.
    """
    folder = Path(folder).expanduser().resolve()
    originals = Path(project.dirs['originals']).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    if folder == originals or folder in originals.parents or originals in folder.parents:
        raise ValueError('Use an incoming folder separate from the project originals folder')
    if poll_interval <= 0 or settle_seconds < 0:
        raise ValueError('poll_interval must be positive; settle_seconds must be non-negative')
    template = project.overview.loc[template_ind].copy()
    if not isinstance(template.get('roi'), str):
        raise ValueError('Configure the template ROI before watching')
    resolution = tuple(literal_eval(template['resolution']))
    previous, handled = {}, {}
    lineprint(f'Watching {folder} using overview row {template_ind}; Ctrl+C to stop.')
    try:
        while True:
            current = {}
            for source in sorted(folder.iterdir()):
                if source.suffix.lower() != '.mp4' or not source.is_file() or source.name.startswith('.'):
                    continue
                try:
                    stat = source.stat()
                    signature = (stat.st_size, stat.st_mtime_ns)
                    current[source] = signature
                    if (not stat.st_size or previous.get(source) != signature
                            or time.time() - stat.st_mtime < settle_seconds
                            or handled.get(source) == signature):
                        continue
                    # A failed signature is reported once; restart the watcher to retry it.
                    handled[source] = signature
                    destination = originals / source.name
                    existing = project.overview.index[project.overview.video == source.stem].tolist()
                    if existing:
                        if (Path(project.dirs['tracked']) / (source.stem + '.csv')).exists():
                            continue
                        raise ValueError('Video name already registered; retry it with AT.track()')
                    if destination.exists():
                        raise FileExistsError(f'Project already contains {destination.name}')
                    cap = cv2.VideoCapture(str(source))
                    try:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        readable, _ = cap.read()
                    finally:
                        cap.release()
                    if not readable or fps <= 0 or frames < 1:
                        raise ValueError('Video is not readable; ensure recording/copying has finished')
                    if (width, height) != resolution:
                        raise ValueError('Video dimensions differ from template; shared geometry would be invalid')
                    temporary = destination.with_name('.' + destination.name + '.incoming')
                    try:
                        shutil.copy2(source, temporary)
                        after = source.stat()
                        if (after.st_size, after.st_mtime_ns) != signature:
                            handled.pop(source, None)
                            continue
                        os.replace(temporary, destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                    row = template.copy()
                    row['video'] = source.stem
                    row['fps'] = fps
                    row['fcount'] = frames
                    row['frame_start'] = 1
                    row['frame_stop'] = frames
                    row['exclude'] = 0
                    row['region'] = 1
                    if extract_background:
                        bgname = source.stem + '_bg.jpg'
                        background = bg_extract(str(destination), 1, frames, 25)
                        if not cv2.imwrite(str(originals / bgname), background):
                            raise OSError('Could not save extracted background')
                        row['bgimg'] = bgname
                    ind = 0 if project.overview.empty else project.overview.index.max() + 1
                    project.overview = pd.concat([project.overview, row.to_frame().T.set_axis([ind])])
                    project.save(silent=True)
                    lineprint(f'Incoming video registered: {source.name} (row {ind})')
                    if project.track(inds=[ind], folder='originals', overwrite=False) is False:
                        lineprint('Folder watcher stopped after tracking failed or was interrupted; fix and retry with AT.track().')
                        return False
                except Exception as exc:
                    lineprint(f'Incoming video {source.name}: {type(exc).__name__}: {exc}')
            previous = current
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        lineprint('Folder watcher stopped.')
