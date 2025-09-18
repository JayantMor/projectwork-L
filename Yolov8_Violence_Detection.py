import os
import cv2
from ultralytics import YOLO
import torch

# ======== CONFIG ========
LOCAL_WEIGHTS = r"D:\PycharmProject\JayantS_Fight_Dectection\yolo_violence_detection_model.pt"
VIDEO_PATH    = r"D:\PycharmProject\JayantS_Fight_Dectection\3.mp4"
OUT_PATH      = os.path.join(os.path.dirname(VIDEO_PATH), "output_videos", "35_yolov8_boxes_local.mp4")
CONF_THRESH   = 0.4
CODEC         = "mp4v"   # if MP4 won't play, use "XVID" and .avi
DEVICE        = 0 if torch.cuda.is_available() else "cpu"  # force GPU if you have CUDA PyTorch installed
# ========================

# sanity checks
if not os.path.exists(LOCAL_WEIGHTS):
    raise FileNotFoundError(f"Weight file not found at: {LOCAL_WEIGHTS}")
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found at: {VIDEO_PATH}")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

print("[Info] loading YOLOv8 model from local weights…")
model = YOLO(LOCAL_WEIGHTS)

# open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*CODEC)
writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, (w, h))
if not writer.isOpened():
    raise RuntimeError("VideoWriter failed to open. Try CODEC='XVID' and output extension .avi")

print(f"[Info] processing… (device={DEVICE})")
frame_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # run YOLO per-frame
    results = model.predict(source=frame, conf=CONF_THRESH, device=DEVICE, verbose=False)
    res = results[0]

    # draw YOLO boxes/labels on a copy (BGR)
    plotted = res.plot()

    writer.write(plotted)
    frame_idx += 1
    if frame_idx % 60 == 0:
        print(f"[Info] {frame_idx} frames…")

cap.release()
writer.release()
print(f"[Done] wrote: {OUT_PATH}")
