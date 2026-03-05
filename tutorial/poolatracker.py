import atracker
import os

def main():
    # Set the path to your tracking folder
    home_dir = os.path.expanduser("~")
    video_folder = os.path.join(home_dir, "Desktop", "tutorial")

    # Initialize ATracker
    AT = atracker.ATracker(video_folder)
    AT.set_config(overwrite=True)
    AT.track(folder="originals", pools=4)  # Adjust pools as needed

if __name__ == "__main__":
    # This guard is required for multiprocessing to work safely on macOS/Windows
    main()