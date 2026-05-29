from .__version__ import __version__
from .atracker import ATracker
from .standalone import track_video, save_atcache, load_atcache, process_video
from .editor import manual_tracker
from .measure import batch_measure
from .process import Processor
from .visualise import Visualiser

__citation__ = (
    "If you use ATracker in your research, please cite:\n"
    "Jolles JW (2026). ATracker: A flexible Python toolkit for animal tracking.\n"
    "https://doi.org/10.5281/zenodo.18889019"
)
