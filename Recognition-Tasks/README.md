# 🧠 Recognition Tasks — Classical Computer Vision
**Course:** PGDM AI & Data Science — Computer Vision


---

## 📋 Overview

This folder contains implementations of **classical object recognition and detection techniques** using OpenCV and Python. All tasks are implemented in Google Colab notebooks following a step-by-step approach — theory first, then code.

The four tasks cover the complete classical CV pipeline:

| # | Task | Technique | Real-World Use Case |
|---|---|---|---|
| 1 | Logo Template Matching | `TM_CCOEFF_NORMED` | Find Amazon logo on product boxes |
| 2 | Shape Matching | Hu Moments + `matchShapes` | Classify food snacks by shape |
| 3 | Face Detection | Haar Cascade + `detectMultiScale` | Count faces in a group photo |
| 4 | LEGO Brick Detector | Template Matching + Contour Filter | Detect bricks on assembly line |

---

## 📁 File Structure

```
Recognition-Tasks/
│
├── logo_template_matching.ipynb      ← Task 1: Logo detection
├── shape_matching_real_food.ipynb    ← Task 2: Shape classification
├── face_detection_haar.ipynb         ← Task 3: Face detection
├── lego_brick_detector.ipynb         ← Task 4: LEGO brick detection
└── README.md
```

---

## 📓 Task 1 — Logo Template Matching

**File:** `logo_template_matching.ipynb`

### 🔍 Concept
Template matching slides a small **reference image (template)** over every position of a larger **input image** and computes a similarity score at each location. The position with the highest score is where the object is found.

**Method used:** `cv2.TM_CCOEFF_NORMED`
- Returns scores between 0 and 1
- Score closer to 1 = better match
- Most reliable for real-world images with lighting variation

### 🖼️ Images Used
- **Input:** `logo_small.jpeg` — Amazon product boxes (search space)
- **Template:** `logo_big.jpeg` — Clean Amazon logo (what we're looking for)

### 🔢 Step-by-Step Pipeline

| Step | Description | Key Function |
|---|---|---|
| 1 | Import libraries | `cv2`, `numpy`, `matplotlib` |
| 2 | Upload input + template images | `files.upload()` |
| 3 | Visualize both images | `plt.imshow()` |
| 4 | Convert BGR → Grayscale | `cv2.cvtColor(...COLOR_BGR2GRAY)` |
| 5 | Run template matching + heatmap | `cv2.matchTemplate(...TM_CCOEFF_NORMED)` |
| 6 | Find best location + draw box | `cv2.minMaxLoc()` + `cv2.rectangle()` |
| 7 | Save & download result | `cv2.imwrite()` + `files.download()` |

### 📊 Results
- Best match score: **~0.32** (low — expected due to scale/print difference)
- The heatmap visualization clearly shows the brightest spot = best match location
- Result image shows bounding box drawn on the boxes photo

### 💡 Key Insight
The low score (~0.32) demonstrates a **real-world limitation** of template matching:
- Logo on the box is printed at a different scale
- Cardboard texture vs. clean digital logo
- Slight perspective distortion on box surface

> This is why real factory systems use **SIFT/ORB** (scale & rotation invariant features) for robust logo detection.

### ⚠️ Limitations
- ❌ Fails if logo is rotated or scaled differently
- ❌ Sensitive to lighting changes and occlusion
- ✅ Works perfectly when appearance is consistent (controlled environments)

---

## 📓 Task 2 — Shape Matching using Hu Moments

**File:** `shape_matching_real_food.ipynb`

### 🔍 Concept
Hu Moments are **7 mathematical values** computed from a shape's contour that act as a unique fingerprint. They remain **invariant** to:
- Rotation — same value even if shape is rotated
- Scale — same value even if shape is bigger/smaller
- Translation — same value even if shape is in a different position

`cv2.matchShapes()` compares two shapes using these moments.
**Score = 0 means identical, higher = more different** (opposite of template matching!)

### 🖼️ Images Used
- **triangle.jpeg** — Bingo chips (triangular snack) — template
- **circle.jpeg** — Ring snacks (circular) — template
- **squre.jpeg** — Wafer biscuits (square) — template
- **Input:** All 3 preprocessed shapes combined into one 900×300 image

### 🔢 Step-by-Step Pipeline

| Step | Description | Key Function |
|---|---|---|
| 1 | Import libraries | `cv2`, `numpy`, `matplotlib` |
| 2 | Upload 3 food images | `files.upload()` |
| 3 | Smart preprocessing per image | Strategy A: Otsu / Strategy B: HSV mask |
| 4 | Visualize original vs preprocessed | `plt.subplots(2, 3)` |
| 5 | Build combined input image | `np.zeros` + array slicing |
| 6 | Extract template contours | `cv2.findContours` + `max(contourArea)` |
| 7 | Classify using matchShapes | `cv2.matchShapes(CONTOURS_MATCH_I1)` |
| 8 | Draw colored labels on result | `cv2.moments()` → centroid → `putText` |
| 9 | Save & download | `cv2.imwrite()` + `files.download()` |

### 🎨 Smart Preprocessing — Two Strategies

| Strategy | Used For | Why |
|---|---|---|
| **Otsu Thresholding** | circle + square (white background) | Auto-finds best intensity threshold |
| **HSV Color Masking** | triangle (yellow/busy background) | Separates chip color from background by hue |

The triangle image (Bingo ad) had a yellow background that confused grayscale thresholding. HSV color masking isolated the orange-brown chips by their hue range (8–28) and subtracted the yellow background.

### 📊 Results
- All 3 shapes correctly classified: Triangle ✅ Circle ✅ Square ✅
- Scores printed for each comparison — lowest score = correct match

### ⚠️ Limitations
- ❌ Fails on textured or complex objects (people, vehicles)
- ❌ Struggles with overlapping shapes
- ✅ Excellent for simple geometric shapes in industrial settings

---

## 📓 Task 3 — Face Detection using Haar Cascade

**File:** `face_detection_haar.ipynb`

### 🔍 Concept
Viola-Jones (2001) — the first **real-time face detector** on CPU, running at 15 FPS on 2001 hardware with no deep learning — just clever engineering.

**How it works:**
1. Uses thousands of **Haar-like features** — black & white rectangles that detect intensity differences (e.g., "is the eye region darker than the forehead?")
2. Trains a **cascade of weak classifiers** (AdaBoost) — early stages quickly reject non-faces, later stages confirm
3. Runs on a **sliding window + image pyramid** — checks every position at multiple scales

OpenCV ships pre-trained Haar cascade XML files — no training needed, just load and use.

### 🖼️ Images Used
-  group photo with multiple faces uploaded via `files.upload()`

### 🔢 Step-by-Step Pipeline

| Step | Description | Key Function |
|---|---|---|
| 1 | Import libraries | `cv2`, `numpy`, `matplotlib` |
| 2 | Upload group photo | `files.upload()` |
| 3 | Load & visualize original | `plt.imshow()` |
| 4 | Load Haar cascade + grayscale | `cv2.CascadeClassifier(cv2.data.haarcascades + ...)` |
| 5 | Detect faces at all scales | `detectMultiScale(scaleFactor, minNeighbors, minSize)` |
| 6 | Draw bounding boxes + count | `cv2.rectangle()` + `cv2.putText()` |
| 7 | Parameter tuning comparison | Default vs stricter `minNeighbors` side by side |
| 8 | Save & download result | `cv2.imwrite()` + `files.download()` |

### 🧮 Key Parameters Explained

| Parameter | Effect |
|---|---|
| `scaleFactor=1.1` | Image shrinks by 10% per pyramid level — finer search |
| `minNeighbors=5` | Need 5 nearby confirmations to keep a detection — removes false positives |
| `minSize=(40,40)` | Ignore detections smaller than 40×40 px — removes tiny noise |

### 📊 Results
- Correctly detected and labeled all visible frontal faces
- Parameter tuning comparison shows precision vs recall trade-off:
  - Lower `minNeighbors` → more detections but more false positives
  - Higher `minNeighbors` → fewer false positives but may miss some faces

### ⚠️ Limitations
- ❌ Fails on side-facing or heavily tilted faces
- ❌ Sensitive to lighting and occlusion
- ❌ More false positives than deep models (MTCNN, RetinaFace)
- ✅ Still used in low-power IoT cameras and as a pre-filter before deep models

---

## 📓 Task 4 — LEGO Brick Detector

**File:** `lego_brick_detector.ipynb`

### 🔍 Concept
Two-stage detection pipeline combining Template Matching and Contour Filtering to detect LEGO bricks — mimicking a factory vision system checking parts on an assembly line.

**Stage 1 — Template Matching:**
- Slides a reference brick image over the scene
- Finds position with highest similarity score
- Demonstrates limitation when brick colors/angles differ

**Stage 2 — Contour Filtering (Color-based):**
- Converts to HSV → masks each brick color separately
- Combines masks → finds contours
- Filters by area and aspect ratio (bricks are roughly square/rectangular)
- Detects ALL bricks regardless of template appearance

### 🖼️ Images Used
- **Scene:** `lego_scene.png` — full photo with 4 colored LEGO bricks
- **Template:** `lego_template.png` — single brick reference image

### 🔢 Step-by-Step Pipeline

| Step | Description | Key Function |
|---|---|---|
| 1 | Import libraries | `cv2`, `numpy`, `matplotlib` |
| 2 | Auto-download LEGO images | `urllib.request.urlretrieve` |
| 3 | Load & visualize both images | `plt.subplots(1, 2)` |
| 4 | Convert to grayscale | `cv2.cvtColor(...COLOR_BGR2GRAY)` |
| 5 | Stage 1: Template matching + NMS | `cv2.matchTemplate` + `cv2.dnn.NMSBoxes` |
| 6 | Draw template matching results | `cv2.rectangle()` + `cv2.putText()` |
| 7 | Stage 2: HSV color masking + contours | `cv2.inRange` + `cv2.findContours` |
| 8 | Compare both stages side by side | `plt.subplots(1, 2)` |
| 9 | Save & download both results | `cv2.imwrite()` + `files.download()` |

### 📊 Results

| Stage | Method | Result |
|---|---|---|
| Stage 1 | Template Matching | Best score: 0.24 — low (template ≠ scene bricks visually) |
| Stage 2 | Color Contour Filter | All 4 bricks detected correctly ✅ |

### 💡 Key Insight — Why Stage 1 Fails Here
Template matching score is low (0.24) because:
- Template brick has different color/angle than scene bricks
- No rotation invariance — template must look identical to target
- This is an honest, real-world demonstration of the limitation

Stage 2 solves this by using **color (HSV)** instead of appearance — doesn't need a matching template.

### ⚠️ Limitations
- ❌ Template matching fails on rotation/scale differences
- ❌ Contour filter may overdetect non-brick rectangular objects
- ✅ Combined approach gives robustness in controlled factory lighting

---

## 🧰 Libraries Used

```python
import cv2          # OpenCV — core computer vision operations
import numpy as np  # Array operations and image manipulation
import matplotlib.pyplot as plt  # Visualization and plotting
```

---

## 🔑 Key OpenCV Functions — Quick Reference

| Function | Task Used In | Purpose |
|---|---|---|
| `cv2.matchTemplate()` | Task 1, 4 | Slide template over image, score each position |
| `cv2.minMaxLoc()` | Task 1, 4 | Find location of best match score |
| `cv2.matchShapes()` | Task 2 | Compare two contours using Hu Moments |
| `cv2.findContours()` | Task 2, 4 | Extract shape outlines from binary image |
| `cv2.CascadeClassifier()` | Task 3 | Load pre-trained Haar cascade model |
| `cv2.detectMultiScale()` | Task 3 | Detect objects at multiple scales |
| `cv2.inRange()` | Task 2, 4 | Color masking in HSV space |
| `cv2.dnn.NMSBoxes()` | Task 4 | Remove duplicate overlapping detections |
| `cv2.morphologyEx()` | Task 2, 4 | Clean binary masks (close holes, remove noise) |
| `cv2.threshold()` | Task 2 | Convert grayscale to binary |

---

## 📊 Comparison — Classical vs Deep Learning

| Aspect | Classical (this implementation) | Deep Learning |
|---|---|---|
| Training needed | ❌ No | ✅ Yes (large dataset) |
| Speed | ✅ Very fast | Depends on hardware |
| Rotation invariance | ❌ Limited | ✅ Yes |
| Generalisation | ❌ Limited to specific appearance | ✅ Robust to variations |
| Explainability | ✅ Fully interpretable | ❌ Black box |
| Best for | Controlled environments, simple shapes | Complex real-world scenes |

---

## 👩‍💻 Author
**Priya A.**
PGDM AI & Data Science | Computer Vision
