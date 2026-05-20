import cv2
import os

os.makedirs("frames", exist_ok=True)

cap = cv2.VideoCapture("Trim.mp4")

i = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
   
    if i % 2 == 0 and saved < 30:
        cv2.imwrite(f"frames/frame_{saved:04d}.png", frame)
        saved += 1
    i += 1

cap.release()
print(f"{saved} frames extraites dans 'frames/'")