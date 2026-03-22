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

- **Encoder** → Extracts features (downsampling)
- **Bottleneck** → Deepest compressed representation
- **Decoder** → Reconstructs spatial info (upsampling)
- **Skip Connections** → Pass encoder features directly to decoder
  → Preserves fine details that would otherwise be lost!

Originally designed for medical image segmentation (2015).
Now used for satellite imagery, autonomous driving, and more.

### Dataset — Oxford-IIIT Pet Dataset
- 37 breeds of cats and dogs
- 7,349 images with pixel-wise segmentation masks
- 3 classes: pet body (1), background (2), border (3)
- Downloads automatically via TensorFlow Datasets

### Model Architecture

| Layer | Operation | Output Size |
|---|---|---|
| Encoder Block 1 | Conv2D(64) × 2 + MaxPool | 64×64 |
| Encoder Block 2 | Conv2D(128) × 2 + MaxPool | 32×32 |
| Encoder Block 3 | Conv2D(256) × 2 + MaxPool | 16×16 |
| Bottleneck | Conv2D(512) × 2 | 16×16 |
| Decoder Block 1 | UpSample + Conv2D(256) × 2 | 32×32 |
| Decoder Block 2 | UpSample + Conv2D(128) × 2 | 64×64 |
| Decoder Block 3 | UpSample + Conv2D(64) × 2 | 128×128 |
| Output | Conv2D(3, softmax) | 128×128×3 |

### Loss Function — Dice Loss
Standard cross-entropy does not work well for segmentation.
Dice Loss measures overlap between prediction and ground truth:

Dice = 2 × |Prediction ∩ Ground Truth| / |Prediction| + |Ground Truth|

- Dice = 1.0 → Perfect segmentation
- Dice = 0.0 → No overlap at all

### Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| IoU (Jaccard) | Intersection / Union | Overlap quality |
| Dice Score | 2×Intersection / (P+GT) | Segmentation accuracy |
| Pixel Accuracy | Correct pixels / Total | Overall accuracy |

---

## 🎯 Practical 4 — YOLOv8 Object Detection

### What is Object Detection?
Object detection finds objects in an image and draws
bounding boxes around them with class labels and confidence scores.

Output format: [x, y, width, height, class, confidence]

### What is YOLO?
**YOLO = You Only Look Once**

Traditional detectors scan image multiple times (slow).
YOLO processes entire image in ONE forward pass → extremely fast!

Evolution:
- YOLOv1 (2016) → First real-time detector
- YOLOv5 (2020) → Most popular version
- YOLOv8 (2023) → Latest, most accurate version

### YOLOv8 Architecture
- **Backbone** → CSPDarknet (feature extraction)
- **Neck** → PANet (multi-scale feature fusion)
- **Head** → Decoupled detection head

### Dataset — COCO (Pre-trained)
- 80 object classes
- 330,000 images
- 1.5 million object instances
- Classes include: person, car, truck, bus, dog, cat,
  bicycle, motorcycle, airplane, bottle, chair and more!

### Model Used
- `yolov8n.pt` → YOLOv8 Nano (fastest, smallest)
- Pre-trained on COCO dataset
- No training needed → use directly for inference!

### Detection Output
For each detected object:
- Bounding box coordinates
- Class label (e.g., "car", "person")
- Confidence score (0.0 to 1.0)

---

## 📊 U-Net vs YOLOv8 Comparison

| Aspect | U-Net | YOLOv8 |
|---|---|---|
| Task | Pixel-wise segmentation | Bounding box detection |
| Output | Segmentation mask | Boxes + labels + scores |
| Speed | Moderate | ⚡ Real-time |
| Precision | Pixel-level | Object-level |
| Training | Trained from scratch | Pre-trained weights |
| Dataset | Oxford-IIIT Pet | COCO (80 classes) |
| Use Case | Medical, satellite | Surveillance, robotics |
| Loss | Dice Loss | CIoU Loss |

---

## 📓 Notebook Structure

| Cell | Description |
|---|---|
| Cell 1-2 | Install & import all libraries |
| Cell 3-4 | Build complete U-Net architecture |
| Cell 5-6 | Define Dice Loss & compile model |
| Cell 7-8 | Load Oxford-IIIT Pet Dataset |
| Cell 9-10 | Train U-Net model |
| Cell 11-12 | Visualize segmentation predictions |
| Cell 13-14 | Calculate IoU & Dice Score |
| Cell 15-16 | Plot training loss curves |
| Cell 17-18 | Install YOLOv8 & load pre-trained model |
| Cell 19-20 | Run YOLOv8 on real-world image |
| Cell 21-22 | Detect objects on multiple images |
| Cell 23-24 | Final Lab 2 summary |

---

## 🌟 Key Learnings

1. **U-Net skip connections** preserve spatial information lost during downsampling
2. **Dice Loss** is better than cross-entropy for imbalanced segmentation tasks
3. **IoU score** is the standard metric for both detection and segmentation
4. **YOLOv8** can detect 80 classes in real-time with high accuracy
5. **Transfer learning** in YOLOv8 saves weeks of training time
6. **Segmentation vs Detection** — choose based on how precise you need the output

---

## 🛠️ Libraries Used

- TensorFlow 2.x / Keras
- TensorFlow Datasets (Oxford-IIIT Pet)
- Ultralytics (YOLOv8)
- NumPy
- Matplotlib
- OpenCV





**Priya A**


AI & Data Science — Computer Vision 



