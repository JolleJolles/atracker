import os 
import cv2
import random
import numpy as np

# Define the video path
video_path = '/Volumes/JOL1_4TB/1601 Experiment sociability interaction rules/Pairtrials/0originals/CM1160116_1130_RP09_S06_G24.mp4'

# Attempt to open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print(f"Error: Cannot open video file at path: {video_path}")
    exit()

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set the number of frames to sample
num_frames_to_sample = 500

# Ensure we don't sample more frames than available
num_frames_to_sample = min(num_frames_to_sample, total_frames)

# Randomly select 500 unique frame indices
random_frame_indices = sorted(random.sample(range(total_frames), num_frames_to_sample))

# Initialize variables for background calculation
ret, frame = cap.read()
height, width, layers = frame.shape
avg_frame = np.zeros((height, width, 3), dtype=np.float32)

# Calculate the average background using the randomly selected frames
for i, frame_idx in enumerate(random_frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        continue
    avg_frame += frame.astype(np.float32) / num_frames_to_sample
    if (i + 1) % 100 == 0:
        print(f"Processing frame {i + 1}/{num_frames_to_sample} for background calculation...")

avg_frame = avg_frame.astype(np.uint8)


# Create an image to accumulate changes
accumulated_changes = np.zeros((height, width, 3), dtype=np.uint8)+255

# Subtract background and accumulate changes for the entire video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_number = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Subtract the background from the current frame
    diff = cv2.absdiff(frame, avg_frame)

    # Convert to grayscale to find the moving object
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to eliminate minor differences
    _, thresh_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    # Optionally, apply morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    thresh_diff = cv2.morphologyEx(thresh_diff, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask_3ch = cv2.cvtColor(thresh_diff, cv2.COLOR_GRAY2BGR)

    # Add the mask to the accumulated changes
    colored_moving_parts = cv2.bitwise_and(frame, mask_3ch)
    non_black_mask = np.any(colored_moving_parts != 0, axis=-1)
    accumulated_changes[non_black_mask] = colored_moving_parts[non_black_mask]

    frame_number += 1
    if frame_number % 100 == 0:
        print(f"Processing frame {frame_number}/{total_frames} for movement accumulation...")

# Combine the background with the accumulated changes
#result = cv2.addWeighted(avg_frame, 0.5, accumulated_changes, 0.5, 0)

# Save the result to the Desktop
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'fish_path_vid3.png')
cv2.imwrite(desktop_path, accumulated_changes)

cv2.destroyAllWindows()
cap.release()