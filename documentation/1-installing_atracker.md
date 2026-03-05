# ATracker Setup Guide (macOS and Windows)

**ATracker** is not yet publicly available through pip or GitHub. Until it is, you can install it manually. This guide covers the basics of setting up a Python environment, installing dependencies, and configuring tools so that you can run ATracker easily on macOS and Windows.

## 1. Terminal and Editor

### Using the terminal

We will make use of the terminal for installing **atracker** and its dependencies:
- **macOS:** Open **Terminal** from **Applications → Utilities → Terminal**.  
- **Windows:** Open **Command Prompt** (type “cmd” in Start menu)

> The terminal/command prompt is a text-based interface where you type commands (like installing Python packages). On **macOS**, you’ll see a prompt like `MacBook:~ user$`. On **Windows**, you might see `C:\Users\YourName>`. From now on "the terminal".

### Editing files

Sometimes you need to edit configuration files directly from the terminal. A simple and commonly used text editor is **nano**. For files that require administrative permissions, use **sudo nano**. For example:
    
```bash
sudo nano ~/.zshrc
```

After making changes in nano:

- Press `Ctrl+X` to exit.
- When prompted "Save modified buffer?", press `Y` to confirm.
- Then press `Enter` to save the file with the current name.

This process will update the file and return you to the terminal prompt.

## 2. Install Python 3.9

**Mac/Linux**

First, install Homebrew (if you haven’t already). In Terminal, paste:

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Then use homebrew to install python:

    brew install python@3.9

After installation, check that Python and pip are available:

    python3 --version
    pip3 --version

> You should see something like Python 3.9.x and pip 21.x or similar.

**Windows**

1. Go to: https://www.python.org/downloads/release/python-390/
2. Scroll  to the **"Files"** section and download **Windows installer (64-bit)** --> e.g., `python-3.9.0-amd64.exe`.

Then, 

1. Run the installer.
2. **IMPORTANT**: Tick the checkbox **"Add Python 3.9 to PATH"* before clicking *Install Now*.
3. In  "Advanced Options", also tick:
    - "Add Python to environment variables"
    - "Precompile standard library"

After installation, check it worked:

    python --version
    pip --version

### 3. Create and Activate a Virtual Environment

✅ Python's built-in `venv` lets you safely install packages without affecting your system Python.

To create a virtual environment on Mac, create the virtual environment as follows. 

First, look for the directory where python3.9 is installed (see your installation log). This is typically:

- `/usr/local/opt/python@3.9/libexec/bin/python` (Intel Mac, Homebrew)
- `/opt/homebrew/opt/python@3.9/bin/python3.9` (Apple Silicon/M1/M2, Homebrew)

Now, create the virtual environment by running (adjust the path as needed):

    /opt/homebrew/opt/python@3.9/bin/python3.9 -m venv ~/.virtualenvs/tracker

or

    /usr/local/opt/python@3.9/libexec/bin/python -m venv ~/.virtualenvs/tracker

And activate it (every time you start a new terminal) with:

    source ~/.virtualenvs/tracker/bin/activate

> If you're on an Intel Mac, use `/usr/local/opt/python@3.9/bin/python3.9` instead.

On Windows, use:

    python -m venv tracker
    tracker\Scripts\activate

You’ll know it’s activated when your terminal prompt shows:

    (tracker) $

To deactivate later:

    deactivate

---

#### Optional: **Set up virtualenvwrapper for easier environment activation**

*This step is optional, but recommended if you want to quickly activate your environment with a simple command (`workon tracker`) instead of typing the full `source ...` path each time.*

1. **Install virtualenvwrapper (if not already):**

    ```bash
    pip3 install virtualenvwrapper
    ```

2. **Add these lines to the end of your `~/.zshrc` (or `~/.bashrc` if using bash):**

    ```bash
    export WORKON_HOME=$HOME/.virtualenvs
    export VIRTUALENVWRAPPER_PYTHON=$(which python3)
    source /usr/local/bin/virtualenvwrapper.sh
    ```

    > *If `/usr/local/bin/virtualenvwrapper.sh` does not exist, run `which virtualenvwrapper.sh` to find the path.*

3. **Restart your terminal or run:**

    ```bash
    source ~/.zshrc
    ```

4. **Now, you can activate your environment at any time with:**

    ```bash
    workon tracker
    ```

    Your prompt should show `(tracker) $`.

5. **To deactivate:**

    ```bash
    deactivate
    ```

## 4. Install ATracker

1. Download or clone the `ATracker` folder to your computer. 
2. Open a terminal folder and make sure your virtual environment is activated.
3. and `cd` into that folder (e.g. on mac `cd ~/Desktop/atracker/`) by typing `cd` on mac or `cd /d` on windows, followed by a space, and dragging the folder into the terminal, and pressing `Enter`.
4. Install `ATracker` in editable mode by typing the following:

    ```bash
    pip install -e
    ```
    
    Make sure there’s a space after `-e`, then drag the `ATracker` folder into the terminal so the full path appears and press Enter.

    > **Tip**: If you use pip install -e , make sure not to move or delete the ATracker folder afterward, as this installation method links directly to the source code. If you want a permanent installation that doesn’t depend on the folder location, use pip install . instead.

## 5. Installing FFmpeg (optional but recommended)

ATracker can convert `.h264` video files (commonly recorded by Raspberry Pi cameras) to `.mp4` format. This functionality works out of the box using a built-in fallback via `imageio[ffmpeg]`, so **you do not need to install FFmpeg manually** for basic use.

However, if you **install FFmpeg on your system**, ATracker will use it automatically for conversion. This is much **faster**, especially for larger video files, because it avoids re-encoding and directly repackages the video. To do this, simply open a new terminal window (make sure you are not in the virtual environment) and run:

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
```bash
winget install "FFmpeg (Essentials Build)"
```

🔍 Verify it's installed:

```bash
ffmpeg -version
```

If it prints version info, you're all set.

> 💡 If FFmpeg is not found, ATracker will still convert videos using imageio, but this may be significantly slower as it decodes and re-encodes each frame.

## 6. Using Your Environment in VS Code or Jupyter

I recommend using **Visual Studio Code (VS Code)** for editing, running, and testing your scripts and notebooks.

1. **Install VS Code:**  

Download from https://code.visualstudio.com/, install, and launch.

2. **Install the Python and Jupyter Extensions:**  

Click the **Extensions icon (left bar)**, then:
    - Install **Python** (by Microsoft)
    - Install **Jupyter** (by Microsoft)

    > This allows you to open .ipynb notebooks and run code interactively.

3. **Install `ipykernel` into your virtual environment:**

Make sure the environment is activated, then run:
    ```bash
    pip install ipykernel
    python -m ipykernel install --user --name tracker --display-name "Python 3.9 (tracker)"
    ```
    > This registers your virtual environment as a selectable "kernel" in Jupyter or VS Code.

    > You can change "Python 3.9 (tracker)" to anything you want — it’s just the name that shows up in menus.


4. **Select interpreter in VS Code:**  
Open the Command Palette:
    - Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (macOS)
    - Type: `Python: Select Interpreter`
    - Choose: **Python 3.9 (tracker)** or browse to your virtualenv’s python.exe

## 6. ATracker Documentation

ATracker’s functions come with built-in documentation strings. You can view the documentation for any function by printing its __doc__ attribute. This works the same on Windows and macOS. 

For example, to display the documentation for the manual tracking function, you can run:

```Python
print(atracker.ATracker.process.__doc__)
```

This will print out the usage information and description for that function directly in your terminal or Python interactive shell.

> **Tip**: If you're using an interactive environment such as Jupyter Notebook in Visual Studio Code, you can also use the question mark operator for quick help:

```Python
atracker.ATracker.process?
```

This command opens a help pane with the documentation for that function.