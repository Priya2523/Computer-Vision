# 🌺 Classical Computer Vision Pipeline
## Lotus Flower Analysis using OpenCV

---

## 📌 What is a Classical CV Pipeline?
A step-by-step image processing workflow where each step
builds on the previous one to extract meaningful information.

Real world use case: Automated flower quality inspection in agriculture!

---

## 🎯 Objective
Apply a complete Classical CV Pipeline on a Lotus flower image to:
- Separate flower from background
- Detect flower boundaries
- Extract shape features
- Make quality decision based on measurements

---

## 📁 Files

| File | Description |
|---|---|
| `classical_computer_vision_pipeline.ipynb` | Main pipeline notebook |
| `lotus_image.jpg` | Input image used |

---

## 🔄 Pipeline Steps

| Step | Operation | Purpose |
|---|---|---|
| Step 1 | Image Acquisition | Load and display original image |
| Step 2 | Grayscale Conversion | Reduce 3 channels to 1 |
| Step 3 | Gaussian Blur | Remove noise before edge detection |
| Step 4 | Canny Edge Detection | Find flower boundaries |
| Step 5 | OTSU Thresholding | Separate flower from dark background |
| Step 6 | Morphological Operations | Clean mask (remove noise, fill holes) |
| Step 7 | Contour Detection | Find flower outline |
| Step 8 | Feature Extraction | Measure area, perimeter, circularity |

---

## 📓 Notebook Structure

| Cell | Description |
|---|---|
| Cell 1-2 | Import libraries |
| Cell 3-4 | Image Acquisition |
| Cell 5-6 | Grayscale Conversion |
| Cell 7-8 | Gaussian Blur |
| Cell 9-10 | Canny Edge Detection |
| Cell 11-12 | OTSU Thresholding |
| Cell 13-14 | Morphological Operations |
| Cell 15-16 | Contour Detection |
| Cell 17-18 | Feature Extraction |
| Cell 19-20 | Full Pipeline Visualization |
| Cell 21-22 | Decision & Final Summary |

---

## 📊 Features Extracted

| Feature | Meaning |
|---|---|
| Area | Total flower size in pixels² |
| Perimeter | Length of flower boundary |
| Circularity | How round the flower is (1.0 = perfect circle) |
| Aspect Ratio | Width to height ratio |
| Area Coverage | % of image occupied by flower |
| Bounding Box | Smallest rectangle around flower |

---

## 💡 Why Lotus Image?
- Dark background → easy segmentation
- Clear petal edges → edge detection works perfectly
- Distinct shape → contour detection is accurate
- High contrast → thresholding works beautifully
- Unique choice — stands out from generic examples!

---

## 🛠️ Libraries Used

- OpenCV (cv2)
- NumPy
- Matplotlib

**Priya A**
- AI & Data Science 

