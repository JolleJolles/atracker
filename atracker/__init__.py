from .__version__ import __version__
from .atracker import ATracker
from .standalone import track_video
from .visual_editor import manual_tracker
from .batch_measure import batch_measure
from .post_processor import Processor

__citation__ = (
    "If you use ATracker in your research, please cite:\n"
    "Jolles JW (2026). ATracker: A flexible Python toolkit for animal tracking.\n"
    "https://doi.org/10.5281/zenodo.18889019"
)
