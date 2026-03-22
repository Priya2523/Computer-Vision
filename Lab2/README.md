# 🔬 Lab 2 — Image Segmentation & Object Detection
## U-Net Semantic Segmentation + YOLOv8 Object Detection

---

## 📌 Objective
Build and evaluate two advanced computer vision systems:
- **Practical 3** → Pixel-wise image segmentation using U-Net architecture
- **Practical 4** → Real-time object detection using YOLOv8

---

## 📁 Files

| File | Description |
|---|---|
| `CV_lab2_UNet_YOLO.ipynb` | Both practicals in one notebook |

---

## 🧪 Practical 3 — U-Net Image Segmentation

### What is Image Segmentation?
Unlike object detection which draws bounding boxes,
segmentation labels every single pixel of the image.

Example:
- Pixel belongs to cat → label 1
- Pixel belongs to background → label 0
- Pixel belongs to border → label 2

### What is U-Net?
U-Net is a CNN architecture with a unique shape:
