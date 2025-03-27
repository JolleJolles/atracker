# ATracker Setup Guide (macOS and Windows)

**ATracker** is not yet publicly available through pip or GitHub. Until it is, you can install it manually. This guide covers the basics of setting up a Python environment, installing dependencies, and configuring tools so that you can run ATracker easily on macOS and Windows.

## 1. Terminal and Editor

### Using the terminal

We will make use of the terminal for installing **atracker** and its dependencies:
- **macOS:** Open **Terminal** from **Applications → Utilities → Terminal**.  
- **Windows:** Open **Command Prompt** (type “cmd” in Start menu)

> The terminal/command prompt is a text-based interface where you type commands (like installing Python packages). On **macOS**, you’ll see a prompt like `MacBook:~ user$`. On **Windows**, you might see `C:\Users\YourName>`.

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

## 2. Installation helper software

### 2.1 Homebrew (macOS)

- Open a terminal window and install homebrew:

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew update
    brew upgrade
    ```

    > **Tip**: Homebrew is a package manager on macOS. It makes installing additional software (like Python, ffmpeg) easier. If you see any prompts during installation (like “Press RETURN to continue”), follow them.

### 2.2 winget (Windows)

- Windows 10 and 11 usually have **winget** installed by default. If not, see Microsoft’s docs:  https://github.com/microsoft/winget-cli

    > **Tip**: winget is a command-line package manager for Windows. Once installed, you can type commands like `winget install something` in the Command Prompt.

---

## 3. Installing Python

### macOS (pyenv + pyenv-virtualenv)

1. Install pyenv and pyenv-virtualenv:

    ```bash
    brew install pyenv pyenv-virtualenv
    ```

    > **Tip**: pyenv lets you manage multiple Python versions without messing up the one that came with your computer. The lines you add to your shell file let your terminal “know” about pyenv each time you open a new terminal window.

2. Configure your shell. The default on macOS Catalina and later is **zsh**, on older versions this is **bash**. Now edit the corresponding file (e.g., `~/.zshrc` or `~/.bashrc`) with `sudo nano` and add the following code to the end of the file:

    ```bash
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    ```

   Save and close the file. Then reload the shell to make the changes effective:

    ```bash
    exec $SHELL
    ```

3. Now install Python 3.9:

    ```bash
    pyenv install 3.9
    ```


### Windows (Official Python Installer)

1. Download Python 3.9 from:  
   https://www.python.org/downloads/windows/  
2. During installation, **check** "*Add Python to PATH*".  
3. After install, open Command Prompt and type:

    ```bash
    python --version
    ```

   to confirm it’s installed.

    > **Tip**: Adding Python to PATH means you can type `python` anywhere in the Command Prompt.  If `python --version` shows something like “3.9.x,” you’re good to go.

## 4. Pip and Setuptools

- After Python is installed, ensure `pip` and `setuptools` are up-to-date:

    ```bash
    python -m pip install --upgrade pip setuptools
    ```

    > **Tip**: `pip` is Python’s package manager. It installs additional libraries (like numpy or opencv).  `setuptools` helps install Python packages that have a `setup.py` file.

## 5. Creating a Virtual Environment

### macOS (using pyenv-virtualenv)

1. Create a virtual environment named `py39` (or chose another name):

    ```bash
    pyenv virtualenv 3.9 py39
    ```

2. Activate it:

    ```bash
    pyenv activate py39
    ```

3. To deactivate:

    ```bash
    pyenv deactivate
    ```

    > **Tip:**: A “virtual environment” is like a separate sandbox for Python packages, and everything you do inside the virtual environment stays there so you don’t break your system Python. When activated, you’ll see `(py39)` in your terminal prompt.

### Windows (using venv)

1. In Command Prompt type:

    ```bash
    python -m venv py39
    ```

2. Now activate it:

    ```bash
    py39\Scripts\activate
    ```

3. To deactivate:

    ```bash
    deactivate
    ```

    >**Tip**: If you see `(py39)` in your prompt, you’re inside the virtual environment.  Everything you install with pip now stays in that environment.


## 6. ipykernel (Optional but Recommended)

1. Make sure your virtual environment is active.  
2. Install ipykernel:

    ```bash
    pip install ipykernel
    ```

3. Register it for Jupyter/Microsoft VS Code:

    ```bash
    python -m ipykernel install --user --name py39 --display-name "Python 3.9 (py39)"
    ```

    > **Tip**: This step lets you pick your environment (“py39”) as a kernel in Jupyter notebooks or VS Code. “Kernel” just means the Python runtime that executes your code. You can choose the display name you want.

## 7. Installing ffmpeg

Next we will install **ffmpeg**. It is optional but in most cases you will need it as you will likely have `.h264` video files that need to be converted (to `.mp4`).

### macOS (Homebrew)

```bash
brew install ffmpeg
```

### Windows (winget)

```bash
winget install "FFmpeg (Essentials Build)"
```

Test by running:

```bash
ffmpeg -version
```

If it prints version info, you’re set.

## 8. Installing ATracker

Now we are ready to install **atracker**! 

1. Download or clone the ATracker package folder to your computer. 
2. In the terminal, `cd` into that folder (e.g. on mac `cd ~/Desktop/atracker/`). You can do that by typing `cd` on mac or `cd /d` on windows, followed by a space, and dragging the folder into the terminal, and pressing `Enter`.
3. Now to install simply run:

    ```bash
    pip install .
    ```

    > **Tip**:This tells pip to look at the current directory (the `.`) and install whatever package is defined there.

4. Finally, if wanting to keep using the most up-to-date version when it becomes available, we actually want to uninstall atracker to be able to load the latest files locally, such as when you replaced certain files of atracker for a newer version. To do so, open temrinal and run:

   ```bash
   pip uninstall atracker -y
   ```

## 9. Using Your Environment in Jupyter and Visual Studio Code

I recommend using **Microsoft Visual Studio Code (VS Code)**. VS Code is a modern code editor that combines a powerful code editor with an integrated terminal and built-in support for Jupyter notebooks. This makes it especially useful for both beginners and advanced users. 

1. **Download and Install VS Code:**  
   - Go to [https://code.visualstudio.com/](https://code.visualstudio.com/) and download the installer for your operating system.  
   - Run the installer and follow the instructions.

2. **Install the Python and Jupyter Extensions:**  
   - Open VS Code.  
   - Click on the Extensions icon on the left sidebar (or press `Ctrl+Shift+X` on Windows or `Cmd+Shift+X` on macOS).  
   - In the search box, type **"Python"** and install the extension by Microsoft.  
   - Then, search for **"Jupyter"** and install the Jupyter extension by Microsoft.  

    > **Tip**: The Jupyter extension lets you run notebooks directly within VS Code, making it easy to interact with your code and see results immediately.

3. **Select the Correct Python Interpreter:**  
   - Open the Command Palette by pressing `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (macOS).  
   - Type **"Python: Select Interpreter"** and choose it.  
   - Select the interpreter that corresponds to your virtual environment (e.g., `Python 3.9 (py39)`).
   - If your virtual environment is not listed, click on **"Enter interpreter path..."** and then **"Find..."**.  
   Navigate to your virtual environment’s `python.exe` (e.g., `<your-project-folder>\py39\Scripts\python.exe`) and select it.

## 10 ATracker Documentation

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