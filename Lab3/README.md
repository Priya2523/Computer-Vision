# 🧠 Lab 3 - CNN Padding, Strides & Object Detection

Experiments on CNN architecture parameters and real-world object detection using deep learning models.

---

## 📌 Lab Objectives

- Understand how **padding** and **strides** affect CNN feature maps
- Compare model accuracy, parameters, and training speed across configurations
- Implement **object detection** using pre-trained deep learning models

---

## 📁 Files

| File | Description |
|------|-------------|
| `lab3_cnn_padding_strides.ipynb` | CNN padding & strides experiments on MNIST |
| `lab3_object_detection.ipynb` | Object detection using pre-trained models |

---

## 🗂️ Notebook 1 — CNN Padding & Strides

### 📦 Cell 1 — Import Libraries
```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np, matplotlib.pyplot as plt
import pandas as pd, time
```
> Imports TensorFlow, Keras, NumPy, Matplotlib, Pandas, and time for experiments.

---

### 📥 Cell 2 — Load & Preprocess MNIST
```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
```
> Loads 70,000 handwritten digit images. Reshapes to (28,28,1) and normalizes to [0,1].

---

### 📖 Cell 3 — Theory: Padding & Strides
```
Padding:
  'valid' → No padding → output shrinks: (28-3)/1 + 1 = 26x26
  'same'  → Zero-pad  → output stays: 28x28

Strides:
  stride=1 → fine-grained, more operations
  stride=2 → coarser, 4x fewer operations
```
> Explains the math behind output size formula before experiments.

---

### ⚙️ Cell 4 — Define 4 Experiment Configurations
```python
configs = [
    {"padding": "valid", "stride": 1, "name": "Valid | Stride 1"},
    {"padding": "same",  "stride": 1, "name": "Same  | Stride 1"},
    {"padding": "valid", "stride": 2, "name": "Valid | Stride 2"},
    {"padding": "same",  "stride": 2, "name": "Same  | Stride 2"},
]
```
> Sets up 4 combinations of padding and stride to compare systematically.

---

### 🔄 Cell 5 — Build & Run All Experiments
```python
# For each config:
# 1. Build CNN model
# 2. Train on MNIST
# 3. Record accuracy, params, time, output size
# 4. Extract feature maps for visualization
```
> Core experiment loop — trains 4 models and logs results for comparison.

---

### 📊 Cell 6 — Summary Table
```python
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
```
> Prints a clean comparison table of all 4 configurations side by side.

---

### 🗺️ Cell 7 — Visualize Feature Maps
```python
# 4 rows (configs) x 5 cols (original + 4 filters)
fig, axes = plt.subplots(4, 5, figsize=(16, 10))
```
> Shows how each padding/stride config transforms the input image through filters.

---

### 📈 Cell 8 — Bar Chart Comparison
```python
# 3 bar charts: Accuracy | Training Time | Parameters
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
```
> Visual comparison of accuracy, speed, and model size across all 4 configs.

---

### 🔍 Cell 9 — Analysis & Observations

| Configuration | Output Size | Effect |
|---|---|---|
| Valid + Stride 1 | 26×26 | Slight edge info loss, high resolution |
| Same + Stride 1 | 28×28 | Preserves all spatial info, more params |
| Valid + Stride 2 | 13×13 | Aggressive downsampling, fastest training |
| Same + Stride 2 | 14×14 | Balanced: downsamples + preserves edges |

---

## 🗂️ Notebook 2 — Object Detection

> Uses pre-trained deep learning models to detect and classify objects in images with bounding boxes and confidence scores.

---

## 🚗 Vehicle Classes (COCO Dataset)
- Person, Car, Bus, Truck, Motorcycle, Bicycle, and 80+ more classes

---

## ⚙️ Installation
```bash
pip install tensorflow keras numpy matplotlib pandas opencv-python
```

---

## 📈 Key Results

- `Same + Stride 1` → **Highest accuracy** (preserves most spatial info)
- `Valid + Stride 2` → **Fastest training** (4x fewer operations)
- `Same + Stride 2` → **Best balance** of speed and accuracy

---

## 👩‍💻 Author

**Priya**  
PGDM AI & Data Science  
Computer Vision — Trimester 3 | Lab 3
