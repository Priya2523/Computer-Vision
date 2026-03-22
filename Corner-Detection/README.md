# 🔍 Corner Detection — Computer Vision
## Harris, Shi-Tomasi & FAST on Mercedes-AMG GT

---

## 📌 What is a Corner?
A corner is a point in an image where brightness changes sharply
in multiple directions simultaneously.

Examples on the Mercedes-AMG GT image:
- Sharp edges of headlights
- Grille mesh intersections
- Mercedes star logo boundary
- Number plate corners
- Windshield edges
- Side mirror boundaries

---

## 🎯 Objective
Implement and compare 3 corner detection algorithms on a
real-world car image and analyze their performance.

---

## 📁 Files

| File | Description |
|---|---|
| `corner_detector.ipynb` | Main notebook |
| `mercedes_benz.jpg` | Input image used |

---

## 🧪 Detectors Implemented

| Detector | Year | Method | Speed |
|---|---|---|---|
| **Harris** | 1988 | Structure tensor + eigenvalue score | Moderate |
| **Shi-Tomasi** | 1994 | Min eigenvalue (improved Harris) | Moderate |
| **FAST** | 2006 | Pixel ring intensity test | ⚡ Very Fast |

---

## 📓 Notebook Structure

| Cell | Description |
|---|---|
| Cell 1-2 | Import libraries |
| Cell 3-4 | Load & display Mercedes image |
| Cell 5-6 | Original vs Grayscale comparison |
| Cell 7-8 | Image gradients (Ix, Iy, Magnitude) |
| Cell 9-10 | Harris Corner Detector + Response Map |
| Cell 11-12 | Harris Sub-pixel Refinement |
| Cell 13-14 | Shi-Tomasi Corner Detector |
| Cell 15-16 | FAST Corner Detector |
| Cell 17-18 | Side-by-side comparison of all 3 |
| Cell 19-20 | Speed test (30 runs each) |
| Cell 21-22 | Robustness test (Gaussian blur) |
| Cell 23-25 | Final comparison bar chart |
| Cell 26-27 | Final summary table |

---

## 🔬 Theory

### Harris Corner Detector
Computes a structure tensor M using image gradients:

R = det(M) - k × trace(M)²

- R >> 0 → Corner ✅
- R < 0  → Edge
- R ≈ 0  → Flat region

Parameters used:
- blockSize = 3 (neighbourhood window)
- ksize = 3 (Sobel kernel)
- k = 0.04 (Harris sensitivity)

### Shi-Tomasi Corner Detector
Improved Harris using minimum eigenvalue:

R = min(λ1, λ2)

More reliable than Harris — avoids false corner detections on edges.

### FAST Corner Detector
No gradient computation — uses pixel ring test:
- Check 16 pixels in a circle around each pixel p
- If 12+ consecutive pixels are brighter OR darker → Corner!
- Extremely fast → used in real-time systems

---

## 📊 Results on Mercedes-AMG GT

| Part of Image | Harris | Shi-Tomasi | FAST |
|---|---|---|---|
| Headlight edges | ✅ Strong | ✅ Strong | ✅ Dense |
| Grille mesh | ✅ Many | ✅ Controlled | ✅ Very Dense |
| Mercedes logo | ✅ Yes | ✅ Yes | ✅ Yes |
| Windshield | ⚠️ Few | ⚠️ Few | ⚠️ Few |
| Background trees | ⚠️ Noisy | ✅ Controlled | ❌ Very Noisy |

---

## ⏱️ Speed Comparison

| Detector | Speed | Best For |
|---|---|---|
| Harris | Moderate | Precise matching, calibration |
| Shi-Tomasi | Moderate | Optical flow, tracking |
| FAST | ⚡ Fastest | Real-time robotics, AR, video |

---

## 🌫️ Robustness (After Gaussian Blur)

Tested all 3 detectors on blurred image to simulate
bad lighting or motion blur conditions.

- Harris → moderate drop in corners
- Shi-Tomasi → controlled, stable results
- FAST → significant drop (sensitive to blur)

---

## 💡 When to Use Which?

- Need **precise corner location** → Harris + cornerSubPix
- Need **best corners for tracking** → Shi-Tomasi
- Need **real-time speed** → FAST

---

## 🛠️ Libraries Used

- OpenCV (cv2)
- NumPy
- Matplotlib
- time (for speed measurement)

---


**Priya A**

**AI & Data Science**

