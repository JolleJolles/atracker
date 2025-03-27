from moviepy import ColorClip

# Video parameters
width = 1000
height = 800
fps = 25
duration = 5 * 60  # 5 minutes in seconds

# Create a white color clip
white_color = (255, 255, 255)
clip = ColorClip(size=(width, height), color=white_color, duration=duration)
clip = clip.set_fps(fps)

# Write the video to a file
clip.write_videofile("tutorial/0originals/white_video.mp4", fps=fps)
