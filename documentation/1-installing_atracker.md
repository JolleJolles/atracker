# Installing ATracker

[Guide overview](2-using_atracker.md)

Install ATracker into a dedicated Python environment, then use it from a script,
Python console or notebook. The package uses OpenCV for video handling and PyQt5
for its interactive editor; interactive workflows need a graphical desktop.

## 1. Install Python and create an environment

ATracker documents Python 3.9+ compatibility. Use a Python installation compatible
with its dependencies; do not replace a working environment just to follow this
example. Python installers are available from [python.org](https://www.python.org/downloads/).
On Windows, enable the installer's option to add Python to PATH.

Open Terminal on macOS/Linux or Command Prompt on Windows. Check your interpreter:

```bash
python3 --version
```

On Windows, use `py --version`. Create and activate an environment:

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows — Command Prompt**

```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

Reactivate the environment when starting a new terminal session. Use `deactivate`
to leave it. Keep the environment outside folders you use for video data.

## 2. Install the package

With the environment active:

```bash
python -m pip install --upgrade pip
python -m pip install git+https://github.com/JolleJolles/atracker.git
```

This installation method requires [Git](https://git-scm.com/downloads). If you have
instead downloaded the source folder, open a terminal in that folder and run:

```bash
python -m pip install .
```

For development, use `python -m pip install -e .` instead. An editable installation
uses the source folder directly, so keep it in place. Python dependencies,
including `pythutils`, are installed from the package's `setup.py` requirements.

Check that the environment can import ATracker:

```bash
python -c "import atracker; print(atracker.__version__); print(atracker.__file__)"
```

## 3. Add FFmpeg if you need video conversion

A system FFmpeg installation is useful for converting camera recordings such as
`.h264` files and for video output. Install it using your system's package manager
or the instructions at [ffmpeg.org](https://ffmpeg.org/download.html). For example,
with Homebrew on macOS:

```bash
brew install ffmpeg
```

Then check it is available:

```bash
ffmpeg -version
```

ATracker also has imageio-based conversion support; a system FFmpeg installation
allows efficient repackaging where appropriate.

## 4. Run scripts or notebooks

For a script, use the activated environment:

```bash
python my_tracking_script.py
```

In VS Code, install the Python extension and select this environment with
**Python: Select Interpreter**. For notebooks, also install the Jupyter extension
and register the environment as a kernel:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name atracker --display-name "Python (ATracker)"
```

Select **Python (ATracker)** as the notebook kernel. After editing package source,
restart the Python session or kernel to load the changes.

## 5. Choose a workflow

Start with the [capabilities and workflow overview](2-using_atracker.md). Use
[folder tracking](3-folder_tracking.md) for a structured project or
[standalone tracking](4-standalone_tracking.md) for individual videos.
Image measurements can be made directly with [measure_images](7-measuring_images.md).

To inspect a function's arguments from Python:

```python
from atracker import track_video
help(track_video)
```

If an import fails, check `python -m pip --version` and the Python/kernel selected
in your editor: installation and execution must use the same environment. GUI
windows need a desktop session. Preserve a working environment's package versions
with `python -m pip freeze > requirements.txt` if you need an installation record.
