# 🚦 Traffic Monitoring & Video Analytics

Real-time vehicle detection, tracking, speed estimation, and congestion detection using **YOLOv8 + OpenCV + Matplotlib**.

---

## 📌 Project Overview

This project implements an intelligent traffic monitoring system that analyzes video footage to:
- Detect and classify vehicles in real-time
- Track individual vehicles across frames
- Estimate vehicle speeds
- Detect traffic congestion automatically
- Generate visual analytics and reports

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|------|---------|
| YOLOv8 (Ultralytics) | Vehicle detection & tracking |
| OpenCV | Video processing & frame manipulation |
| Matplotlib | Analytics charts & visualization |
| Pandas | Data analysis & reporting |
| yt-dlp | Auto-download traffic video |
| Python 3.x | Core programming language |

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `Traffic_Monitoring_Video_Analytics.ipynb` | Main Jupyter notebook with full code |
| `traffic_video.mp4` | Input traffic video |
| `output_traffic.mp4` | Processed output video with detections |
| `yolov8n.pt` | YOLOv8 Nano pre-trained model weights |
| `sample_frames.png` | Sample processed frames from output |
| `traffic_analytics.png` | Analytics charts and statistics |

---

## 🗂️ Notebook Structure

### 📦 Cell 1 — Install & Import Libraries
```python
# Installs required packages
!pip install ultralytics yt-dlp -q

# Imports
import cv2, numpy as np, matplotlib.pyplot as plt
import requests, time, os, pandas as pd
from ultralytics import YOLO
```
> Installs YOLOv8 (ultralytics) and yt-dlp. Imports all necessary libraries for video processing, detection, and visualization.

---

### 🎬 Cell 2 — Download Traffic Video Automatically
```python
# Auto-downloads a real traffic video from YouTube
# Uses yt-dlp to fetch best quality mp4 (max 480p)
# Downloads only first 30 seconds for faster processing
VIDEO_PATH = "traffic_video.mp4"
```
> No manual video upload needed. The notebook fetches a real traffic video automatically using yt-dlp.

---

### ⚙️ Cell 3 — Load YOLOv8 + Setup Parameters
```python
model_yolo = YOLO("yolov8n.pt")

# Vehicle class IDs (COCO dataset)
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Video settings
FPS = cap.get(cv2.CAP_PROP_FPS)
WIDTH = 768
HEIGHT = 432

# Congestion threshold
LOW_SPEED_THRESHOLD = 20
```
> Loads YOLOv8 Nano model. Defines vehicle classes from COCO dataset. Sets video resolution and congestion speed threshold.

---

### 🔄 Cell 4 — Main Processing Loop
```python
# Processes every frame of the video:
# 1. Runs YOLOv8 detection + tracking
# 2. Calculates speed from position difference
# 3. Checks congestion conditions
# 4. Draws bounding boxes, labels, speed on frame
# 5. Saves processed frame to output_traffic.mp4
```
> Core cell — detects vehicles, assigns tracking IDs, estimates speed per vehicle, flags congestion, and writes annotated output video.

---

### 📊 Cell 5 — Video Analytics Charts
```python
df = pd.DataFrame(vehicle_log)
df['time_sec'] = df['frame'] / FPS

# Generates:
# - Vehicle count over time (line chart)
# - Average speed over time (line chart)
# - Congestion events timeline
# - Vehicle type distribution (bar chart)
```
> Converts the vehicle log into a Pandas DataFrame and generates 4 analytics charts saved as `traffic_analytics.png`.

---

### 🖼️ Cell 6 — Sample Frames + Final Summary
```python
# Extracts 6 evenly spaced frames from output video
# Displays them in a 2x3 grid
# Prints final analytics report:
#   - Total frames, Duration, Avg vehicles
#   - Max vehicles, Avg speed, Congestion events
```
> Visual summary of the processed video. Shows sample frames and prints the complete analytics report.

---

## 🚗 Vehicle Classes Detected

- 🚗 Car (Class ID: 2)
- 🏍️ Motorcycle (Class ID: 3)
- 🚌 Bus (Class ID: 5)
- 🚛 Truck (Class ID: 7)

---

## ⚙️ How It Works

1. **Video Input** — Traffic video auto-downloaded via yt-dlp
2. **YOLOv8 Detection** — Each frame processed to detect vehicles
3. **Object Tracking** — Vehicles tracked across frames with unique IDs
4. **Speed Estimation** — Speed calculated from position changes between frames
5. **Congestion Detection** — Triggered when vehicles move below `LOW_SPEED_THRESHOLD = 20`
6. **Analytics** — Charts generated for vehicle count, speed, and congestion over time

---

## 📊 Features

- ✅ Real-time bounding boxes with vehicle labels
- ✅ Speed display per vehicle (px/s)
- ✅ Congestion alert overlay on video
- ✅ Vehicle count per frame
- ✅ Analytics charts (speed, count, congestion timeline)
- ✅ Sample frame extraction from output video
- ✅ Full analytics report printed in output
- ✅ Auto video download (no manual upload needed)

---

## 🔧 Installation & Setup
```bash
pip install ultralytics opencv-python matplotlib pandas yt-dlp
```

Then open and run all cells in `Traffic_Monitoring_Video_Analytics.ipynb` sequentially.

---

## 📈 Sample Output

### Processed Frames
![Sample Frames](sample_frames.png)

### Analytics Charts
![Traffic Analytics](traffic_analytics.png)

---

## 📋 Analytics Report Includes

- Total frames processed
- Video duration (seconds)
- Average & maximum vehicle count
- Average speed across all vehicles
- Congestion detection events

---

## 👩‍💻 Author

**Priya**  
PGDM AI & Data Science  
Computer Vision — Trimester 3
