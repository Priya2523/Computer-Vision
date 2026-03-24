# 🔍 Feature Matching — SIFT, ORB & AKAZE
## Feature Descriptors, Keypoint Matching & Image Acquisition Pipeline

---

## 📌 What is Feature Matching?
Feature matching finds corresponding points between two images.
Used in: panorama stitching, object recognition, 3D reconstruction, AR.

Steps:
1. Detect keypoints in both images
2. Compute descriptors (mathematical fingerprint of each keypoint)
3. Match descriptors between images
4. Filter good matches

---

## 📁 Files

| File | Description |
|---|---|
| `Feature_Descriptors_Matching.ipynb` | SIFT, ORB, AKAZE detection + BFMatcher + FLANN |
| `Image_Acquisition_Pipeline.ipynb` | Full CV pipeline from acquisition to feature extraction |

---

## 🧪 File 1 — Feature Descriptors & Matching

### Algorithms Implemented

| Algorithm | Full Name | Type | Speed | Patent |
|---|---|---|---|---|
| **SIFT** | Scale Invariant Feature Transform | Float descriptor | Moderate | ✅ Free (expired) |
| **ORB** | Oriented FAST + Rotated BRIEF | Binary descriptor | ⚡ Fast | ✅ Free |
| **AKAZE** | Accelerated KAZE | Binary descriptor | Fast | ✅ Free |

> Note: SURF is patented so we use AKAZE as the modern free replacement!

### SIFT — How it Works
1. Build scale-space pyramid (Gaussian blurs at different scales)
2. Find keypoints at scale-space extrema (DoG)
3. Assign orientation based on local gradients
4. Compute 128-dimensional descriptor vector
5. Scale invariant + rotation invariant + partially illumination invariant

### ORB — How it Works
1. Detect keypoints using FAST algorithm
2. Assign orientation using intensity centroid
3. Compute binary descriptor using BRIEF
4. Very fast — ideal for real-time applications
5. Binary descriptor → uses Hamming distance for matching

### AKAZE — How it Works
1. Uses nonlinear diffusion filtering (better than Gaussian)
2. More robust to noise than SIFT
3. Modern replacement for patented SURF
4. Works well on textured surfaces

### Matchers Implemented

| Matcher | Method | Best For |
|---|---|---|
| **BFMatcher** | Compare every descriptor pair | Small datasets, accuracy |
| **FLANN** | Approximate nearest neighbor search | Large datasets, speed |

### Lowe's Ratio Test
After matching, we filter good matches using:
- ratio = distance of best match / distance of second best match
- If ratio < 0.75 → good match (keep it!)
- This removes ambiguous matches

---

## 🖼️ File 2 — Image Acquisition Pipeline

### What is an Image Acquisition Pipeline?
A complete CV workflow from raw image to extracted features:

### Pipeline Steps

| Step | Operation | Purpose |
|---|---|---|
| Step 1 | Image Acquisition | Load/download image |
| Step 2 | Preprocessing | Grayscale + Blur + Histogram Equalization |
| Step 3 | Segmentation | Thresholding + Contour Detection |
| Step 4 | Feature Extraction | ORB Keypoints + Descriptors |
| Step 5 | Feature Matching | BFMatcher between two images |
| Step 6 | Classification | Label regions as Large/Small |
| Step 7 | Post Processing | NMS + Final output |

---

## 📊 Algorithm Comparison

| Feature | SIFT | ORB | AKAZE |
|---|---|---|---|
| Descriptor size | 128 float | 256 bits | 486 bits |
| Speed | Slow | ⚡ Fast | Medium |
| Scale invariant | ✅ Yes | ✅ Yes | ✅ Yes |
| Rotation invariant | ✅ Yes | ✅ Yes | ✅ Yes |
| Noise robust | Medium | Low | ✅ High |
| Patent free | ✅ Yes | ✅ Yes | ✅ Yes |
| Best for | Accuracy | Real-time | Noisy images |

---

## ⏱️ Matcher Speed Comparison

| Matcher | Time complexity | Best for |
|---|---|---|
| BFMatcher | O(n²) | Small image pairs |
| FLANN | O(n log n) | Large scale matching |

---

## 📓 Notebook Structure — Feature Descriptors

| Cell | Description |
|---|---|
| Cell 1-2 | Import libraries |
| Cell 3-4 | Load two images from URL |
| Cell 5-6 | SIFT keypoint detection |
| Cell 7-8 | ORB keypoint detection |
| Cell 9-10 | AKAZE keypoint detection |
| Cell 11-12 | Side by side comparison |
| Cell 13-14 | BFMatcher matching |
| Cell 15-16 | FLANN matcher |
| Cell 17-18 | Speed benchmark |
| Cell 19 | Final summary |

---

## 🛠️ Libraries Used

- OpenCV (cv2)
- NumPy
- Matplotlib
- Requests (for downloading images)


##

**Priya A**
AI & Data Science — Computer Vision Assignment
