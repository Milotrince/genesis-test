import cv2
import numpy as np
import time
from moviepy import ImageSequenceClip

# Create dummy frames (e.g., 100 frames of 640x480 RGB)
num_frames = 100
height, width = 480, 640
fps = 30
frames = [np.random.randint(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(num_frames)]

# --- OpenCV VideoWriter ---
start_cv2 = time.time()
out = cv2.VideoWriter('output_cv2.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
for frame in frames:
    out.write(frame)
out.release()
cv2_time = time.time() - start_cv2

# --- MoviePy ImageSequenceClip ---
start_mp = time.time()
clip = ImageSequenceClip(frames, fps=fps)
clip.write_videofile('output_moviepy.mp4', fps=fps, logger=None, codec='libx264', preset='ultrafast')
mp_time = time.time() - start_mp

print(f"OpenCV VideoWriter time: {cv2_time:.2f} seconds")
print(f"MoviePy ImageSequenceClip time: {mp_time:.2f} seconds")
