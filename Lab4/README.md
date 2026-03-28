# Lab 4— Deep Learning for Visual Recognition

This lab covers two advanced Computer Vision projects built using deep learning:

1. **Facial Recognition System** using a FaceNet-inspired CNN with Triplet Loss
2. **Medical Image Analysis** for Disease Detection using Transfer Learning

Both labs are implemented in Python (Google Colab ) using TensorFlow/Keras.

---

## 🗂️ Files in this Lab

| File | Description |
|------|-------------|
| `Lab4_Facial_Rec_img.ipynb` | End-to-end facial recognition pipeline using simplified FaceNet + triplet loss |
| `Lab4_Medical_analysis.ipynb` | CNN-based disease detection on chest X-rays using ResNet50 transfer learning |

---

---

# 📘 Lab A — Facial Recognition System using FaceNet Architecture

## 🎯 Objective

Build a complete end-to-end facial recognition system using a simplified FaceNet-inspired CNN architecture with Triplet Loss to perform:

- **Face Detection** — detect and crop faces from images/video
- **Embedding Extraction** — generate 128-dimensional face descriptors
- **Face Verification (1:1)** — confirm if two faces belong to the same person
- **Face Identification (1:N)** — identify a person from a known database

---

## 🧠 Why FaceNet Architecture?

FaceNet (Google, 2015) is one of the most influential papers in face recognition. Its core idea is elegant:

> **"Learn an embedding space where the distance between embeddings directly corresponds to face similarity."**

- Same person → embeddings are **close** in vector space
- Different people → embeddings are **far apart**
- Uses **Triplet Loss** instead of softmax — the model learns to compare, not classify
- Output: 128-D **L2-normalized** vector — works with any distance metric (Euclidean / Cosine)
- Scales to millions of identities without retraining
- Still used as a strong baseline in 2026; newer SOTA includes ArcFace, MagFace, InsightFace

This architecture was chosen because it teaches the **fundamentals of metric learning** — a crucial concept beyond simple classification.

---

## 🖼️ Why This Dataset? (LFW — Labeled Faces in the Wild)

The notebook uses the **LFW (Labeled Faces in the Wild)** dataset, loaded via `sklearn.datasets.fetch_lfw_people`.

**Reasons for choosing LFW:**

- ✅ **Publicly available** — no manual download 
- ✅ **Built into scikit-learn** — `fetch_lfw_people()` handles everything automatically
- ✅ **Standard benchmark** — used in virtually every face recognition paper for comparison
- ✅ **Realistic images** — faces captured in unconstrained, real-world conditions 
- ✅ **Multiple images per person** — essential for building triplet pairs (anchor, positive, negative)
- ✅ **Right size for a lab** — manageable (~13,000 images, ~5,749 identities) 

---

## 🔧 Tech Stack

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | Model building, training, triplet loss |
| OpenCV DNN (SSD + ResNet10) | Face detection (more reliable than Haar cascades) |
| scikit-learn (LFW) | Dataset loading |
| NumPy | Array operations, triplet generation |
| Matplotlib | Embedding visualization, training plots |
| DeepFace | Optional face analysis comparison |

---

## 🏗️ Model Architecture

```
Input: 160 × 160 × 3
    │
    ▼
Conv2D(64, 5×5, stride=2) → BatchNorm → ReLU → MaxPool
    │
    ▼
Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool
    │
    ▼
Conv2D(256, 3×3) → BatchNorm → ReLU → MaxPool
    │
    ▼
Conv2D(512, 3×3) → BatchNorm → ReLU
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dense(128)
    │
    ▼
L2 Normalization  ←── embeddings lie on a unit hypersphere
    │
    ▼
Output: 128-D embedding vector
```

---

## 📐 Triplet Loss — How It Works

```
Loss = max( ||f(A) - f(P)||² - ||f(A) - f(N)||² + margin, 0 )
```

| Term | Meaning |
|------|---------|
| `A` (Anchor) | Any image of person X |
| `P` (Positive) | Another image of the **same** person X |
| `N` (Negative) | An image of a **different** person Y |
| `margin` | Minimum enforced gap between positive and negative distances (set to 0.5) |

The model is forced to bring anchor-positive pairs closer while pushing anchor-negative pairs apart — this is **metric learning**.

---

## 📋 Notebook Structure (27 Cells)

| Step | Content |
|------|---------|
| Step 1 | Install libraries & imports |
| Step 2 | Face detection using OpenCV DNN (SSD + ResNet10) |
| Step 3 | Build simplified FaceNet model |
| Step 4 | Triplet Loss implementation |
| Step 5 | Load LFW dataset & build triplet pairs |
| Step 6 | Train the siamese triplet model |
| Step 7 | Evaluate — verification accuracy, t-SNE embedding visualization |
| Step 8 | Real-time demo — webcam face identification |

---

## 📊 Evaluation Metrics

- **Verification Accuracy** — correct same/different person decisions (target: >85%)
- **Identification Accuracy** — correct identity from closed-set database (target: >90%)
- **Cosine Distance Threshold** — tuned between 0.4–0.7
- **t-SNE / UMAP** — 2D visualization of 128-D embedding clusters

---

## 💡 Key Design Decisions

- **OpenCV DNN over MTCNN** — faster inference, no extra dependencies, works in Colab without issues
- **L2 normalization** — all embeddings lie on a unit hypersphere, making cosine similarity equivalent to dot product
- **Dropout(0.5) before embedding layer** — prevents overfitting on small datasets
- **Margin = 0.5** in triplet loss — standard starting value, can be tuned
- **128-D embedding** — original FaceNet uses 128-D; compact, efficient, sufficient for lab scale

---

---

# 📗 Lab B — Medical Image Analysis: CNN for Disease Detection

## 🎯 Objective

Apply Convolutional Neural Networks to analyze medical images and automatically detect disease, with a focus on:

- **Binary classification** — Normal vs. Pneumonia from chest X-rays
- **Transfer learning** — leverage ImageNet-pretrained ResNet50
- **Class imbalance handling** — class weights, augmentation
- **Interpretability** — Grad-CAM heatmaps showing where the model looks
- **Medically relevant evaluation** — recall, specificity, AUC-ROC (not just accuracy)

---

## 🩻 Why This Dataset? (Chest X-Ray — Pneumonia Detection)

The notebook downloads the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle (`paultimot/chest-xray-pneumonia`).

**Reasons for choosing this dataset:**

- ✅ **High clinical relevance** — pneumonia is a leading cause of death worldwide; early detection matters
- ✅ **Real medical images** — actual patient X-rays, not synthetic data
- ✅ **Clear binary task** — Normal vs. Pneumonia, easy to frame and evaluate
- ✅ **Demonstrates class imbalance** — ~74% pneumonia vs ~26% normal — a real-world challenge
- ✅ **Large enough to fine-tune** — ~5,800 training images, enough for transfer learning
- ✅ **Widely cited baseline** — easy to compare results with published papers

**Why X-rays specifically?**
Chest X-rays are the most common diagnostic imaging tool globally — building AI for X-rays has direct, real-world impact. They also have well-defined visual patterns (lung opacity, consolidation) that CNNs can learn effectively.

---

## 🔧 Tech Stack

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | Model building and training |
| Keras Applications (ResNet50) | Pre-trained backbone |
| ImageDataGenerator | Real-time augmentation |
| scikit-learn | Class weights, metrics, confusion matrix |
| OpenCV | Image preprocessing |
| Matplotlib / Seaborn | Training curves, confusion matrix plots |
| Kaggle API | Dataset download |

---

## 🏗️ Model Architecture — Transfer Learning with ResNet50

```
Input: 224 × 224 × 3 (X-ray image)
    │
    ▼
ResNet50 (pre-trained on ImageNet, 25M params)
  [Phase 1: Frozen]
  [Phase 2: Last 40 layers unfrozen for fine-tuning]
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dense(512, activation='relu')
    │
    ▼
BatchNormalization
    │
    ▼
Dropout(0.5)
    │
    ▼
Dense(2, activation='softmax')  ←── Normal / Pneumonia
    │
    ▼
Output: Class probabilities
```

**Two-phase training strategy:**
- **Phase 1** — Freeze ResNet50, train only the head (15 epochs, lr=1e-4)
- **Phase 2** — Unfreeze last 40 layers, fine-tune with very low lr (10 epochs, lr=1e-5)

---

## 📋 Notebook Structure (36 Cells)

| Step | Content |
|------|---------|
| Step 1 | Install libraries & imports |
| Step 2 | Download Chest X-Ray dataset via Kaggle API |
| Step 3 | Organize dataset into `Normal/` and `Pneumonia/` folders |
| Step 4 | Class distribution analysis + sample image visualization |
| Step 5 | Build `ImageDataGenerator` with augmentation (train/val split) |
| Step 6 | Compute class weights for imbalance handling |
| Step 7 | Build ResNet50 transfer learning model |
| Step 8 | Phase 1 training — frozen backbone |
| Step 9 | Phase 2 training — fine-tuning |
| Step 10 | Evaluation — confusion matrix, classification report, AUC |
| Step 11 | Grad-CAM heatmap visualization |
| Step 12 | Error analysis — false negatives inspection |

---

## 📊 Evaluation Metrics (Why Not Just Accuracy?)

In medical imaging, **accuracy alone is misleading** due to class imbalance.

| Metric | Importance in Medical Context |
|--------|-------------------------------|
| **Recall / Sensitivity** | Most critical — missing a disease (false negative) is dangerous |
| **Specificity** | Avoiding unnecessary treatment (false positives) |
| **AUC-ROC** | Overall model discrimination ability, threshold-independent |
| **F1 Score** | Balance between precision and recall |
| **Confusion Matrix** | Full breakdown of prediction types |

> ⚠️ A model that predicts "Pneumonia" for everything gets 74% accuracy but is clinically useless. **Recall is the priority.**

---

## 🔍 Data Augmentation Strategy

```python
ImageDataGenerator(
    rescale        = 1./255,       # Normalize pixel values to [0,1]
    rotation_range = 20,           # Random rotations ±20°
    width_shift    = 0.2,          # Horizontal shift
    height_shift   = 0.2,          # Vertical shift
    zoom_range     = 0.2,          # Random zoom
    horizontal_flip= True,         # Mirror image
    fill_mode      = 'nearest'     # Fill empty pixels
)
```

Augmentation is critical in medical imaging because:
- Datasets are small (privacy, annotation cost)
- Prevents overfitting
- Makes model robust to positioning variations in real scans

---

## 🌡️ Grad-CAM — Model Interpretability

Grad-CAM (Gradient-weighted Class Activation Mapping) produces **heatmaps** showing which regions of the X-ray the model focused on to make its prediction.

- **Why it matters in medicine**: Doctors cannot trust a "black box" — they need to see the model's reasoning
- Highlights lung consolidation regions in pneumonia cases
- Helps identify if the model learned clinical features or dataset artifacts
- Last conv layer used: `conv5_block3_out` (ResNet50)

---

## 💡 Key Design Decisions

- **ResNet50 over EfficientNet** — more interpretable layer naming, easier Grad-CAM extraction, widely understood baseline
- **Class weights over oversampling** — avoids creating duplicate images, handles imbalance mathematically
- **Validation split = 20%** — standard for this dataset size
- **Early stopping** (optional) — prevents overfitting when val_loss stops improving
- **Focal loss as bonus** — for students who want to push further

---

---




## 📂 Folder Structure

```
Lab4/
├── Lab4_Facial_Rec_img.ipynb       # Facial Recognition notebook
├── Lab4_Medical_analysis.ipynb     # Medical Image Analysis notebook
└── README.md                       # This file
```

---

## 🔗 References

- FaceNet: [Schroff et al., 2015](https://arxiv.org/abs/1503.03832)
- LFW Dataset: [Huang et al., UMass](http://vis-www.cs.umass.edu/lfw/)
- Chest X-Ray Dataset: [Kaggle — Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

** Priya A **

** Computer Vision - AI & DS **
