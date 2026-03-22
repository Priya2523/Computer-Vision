# 🧠 Lab 1 — CNN from Scratch & Transfer Learning

---

## 📌 Objective
- Practical 1: Build and train a CNN from scratch on CIFAR-10
- Practical 2: Transfer Learning using MobileNetV2 on Cats vs Dogs

---

## 📁 Files

| File | Description |
|---|---|
| `CV_lab1.ipynb` | Both practicals in one notebook |

---

## 🧪 Practical 1 — CNN from Scratch (CIFAR-10)

### Dataset
- 60,000 colour images (32×32 pixels)
- 10 classes: airplane, automobile, bird, cat, deer,
  dog, frog, horse, ship, truck
- 50,000 train / 10,000 test
- Downloads automatically — no manual download needed!

### Model Architecture
- Block 1: Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
- Block 2: Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
- Dense: Flatten → Dense(512) → BatchNorm → Dropout(0.5) → Output(10, softmax)

### Results

| Metric | Value |
|---|---|
| Epochs | 20 |
| Batch Size | 128 |
| Optimizer | Adam |
| Train Accuracy | 91.15% |
| Test Accuracy | 80.53% |
| Sample Predictions | 14/16 correct (87.5%) |

---

## 🔁 Practical 2 — Transfer Learning (Cats vs Dogs)

### Dataset
- Binary classification: Cats = 0, Dogs = 1
- Loaded via TensorFlow Datasets
- Downloads automatically — no manual download needed!

### Model
- Base: MobileNetV2 (pre-trained on ImageNet 1.4M images)
- Head: GlobalAveragePooling → Dense(128) → Dropout(0.5) → Dense(1, sigmoid)

### Phase 1 — Feature Extraction (Base frozen)

| Metric | Value |
|---|---|
| Epochs | 5 |
| Train Accuracy | 99.37% |
| Val Accuracy | 97.97% |

### Phase 2 — Fine Tuning (Last 20 layers unfrozen)

| Metric | Value |
|---|---|
| Epochs | 5 |
| Train Accuracy | 98.94% |
| Val Accuracy | 97.03% |

---

## 📊 Key Observation

| Approach | Dataset | Accuracy | Epochs |
|---|---|---|---|
| CNN from Scratch | CIFAR-10 | 80.53% | 20 |
| Transfer Learning | Cats vs Dogs | 97.97% | 5 |

Transfer Learning = Higher accuracy + Faster training! 🚀

---

## 📓 Notebook Structure

| Cell | Description |
|---|---|
| 1-2 | Import libraries |
| 3-4 | Load & prepare CIFAR-10 |
| 5-6 | Visualize CIFAR-10 images |
| 7-8 | Build CNN model |
| 9-10 | Compile model |
| 11-12 | Train model (20 epochs) |
| 13-14 | Evaluate on test set |
| 15-16 | Plot accuracy & loss curves |
| 17-18 | Predictions on test images |
| 19-20 | Practical 1 summary |
| 21-22 | Practical 2 imports |
| 23-24 | Load Cats vs Dogs dataset |
| 25-26 | Visualize dataset |
| 27-28 | Load MobileNetV2 base |
| 29-30 | Add classification head |
| 31-32 | Phase 1 training |
| 33-34 | Phase 2 fine tuning |
| 35-36 | Plot training curves |
| 37-38 | Predictions on images |
| 39-40 | Final summary |

---

## 🛠️ Libraries Used

- TensorFlow 2.x / Keras
- MobileNetV2 (ImageNet weights)
- TensorFlow Datasets
- NumPy
- Matplotlib


**Priya A**
**AI & Data Science — Computer Vision **
