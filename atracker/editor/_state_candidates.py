"""CSV event windows shared with MoveR; no coordinate files are changed."""
import os
import re
import tempfile
from pathlib import Path

import pandas as pd


class StateCandidates:
    def __init__(self, csv_file, video_dir, frame_base=1, modifier=None):
        if frame_base not in (0, 1):
            raise ValueError("frame_base must be 0 or 1")
        self.path = Path(csv_file).expanduser().resolve()
        self.frame_base = frame_base
        self.source = self.path.read_bytes()
        self.data = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        required = {"video", "start_frame", "end_frame", "state"}
        if not required.issubset(self.data.columns):
            raise ValueError("Candidate CSV requires video, start_frame, end_frame and state")
        if self.data.empty:
            raise ValueError("Candidate CSV contains no events")
        for column in ("video", "state"):
            if self.data[column].str.strip().eq("").any():
                raise ValueError(f"{column} cannot be empty")
        for column in ("start_frame", "end_frame"):
            values = pd.to_numeric(self.data[column], errors="raise")
            if ((values % 1 != 0) | (values < frame_base) | values.isna()).any():
                raise ValueError(f"{column} must contain whole frames >= {frame_base}")
            self.data[column] = values.astype("int64")
        if (self.data.end_frame < self.data.start_frame).any():
            raise ValueError("end_frame cannot precede start_frame")
        if "review" not in self.data:
            self.data["review"] = ""
        if modifier is not None:
            if not isinstance(modifier, str) or not modifier.strip():
                raise ValueError("modifier must be a non-empty column name")
            modifier = modifier.strip()
            if modifier in required | {"review"}:
                raise ValueError("modifier cannot replace a required event or review column")
            if modifier not in self.data:
                self.data[modifier] = ""
            if not self.data[modifier].str.lower().isin(["", "true", "false", "1", "0"]).all():
                raise ValueError(f"Modifier {modifier!r} must contain blank, true or false values")
        self.modifier = modifier
        self.saved = self.data.copy(deep=True)
        root = Path(video_dir).expanduser().resolve()
        self.paths = {}
        media = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in
                 (".mp4", ".avi", ".mov", ".mkv", ".m4v")]
        for video in self.data.video.unique():
            name = Path(video).name
            matches = [p for p in media if p.name == name or p.stem == name]
            if not matches:
                # Region-specific MoveR names still use the original full video.
                stem = re.sub(r"_R\d+$", "", Path(name).stem if Path(name).suffix else name)
                matches = [p for p in media if p.stem == stem]
            if len(matches) != 1:
                raise ValueError(f"Expected one original video for {video!r}; found {len(matches)} in {root}")
            self.paths[video] = str(matches[0])

    @property
    def dirty(self):
        return not self.data.equals(self.saved)

    def update(self, index, first, last, review=None):
        if first < self.frame_base or last < first:
            raise ValueError("Start must not exceed end, and both frames must be within the video")
        self.data.loc[index, ["start_frame", "end_frame"]] = [int(first), int(last)]
        if review is not None:
            self.data.loc[index, "review"] = review

    def save(self):
        if self.path.read_bytes() != self.source:
            raise RuntimeError("The CSV changed outside this window. Reopen it before saving to avoid losing edits.")
        fd, name = tempfile.mkstemp(prefix=".state-review-", suffix=".csv", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
                self.data.to_csv(stream, index=False)
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        self.source = self.path.read_bytes()
        self.saved = self.data.copy(deep=True)

    def file_infos(self):
        offset = 1 - self.frame_base
        return [dict(video_path=self.paths[row.video], video_name=row.video,
                     frame_start=int(row.start_frame) + offset,
                     frame_stop=int(row.end_frame) + offset)
                for row in self.data.itertuples()]
