# 🚗 Tesla Full Self-Driving — Computer Vision Case Study



**Subject:** Computer Vision in Autonomous Vehicles
**Company:** Tesla, Inc.
**System:** Full Self-Driving (FSD) — HydraNet Architecture
**Sensors:** 8 cameras · No radar · No LiDAR

---

## 📌 Overview

This case study examines how Tesla built one of the most ambitious
computer vision systems in the world — a camera-only autonomous
driving pipeline that processes **2 GB of visual data per second**,
runs **50+ parallel vision tasks**, and translates raw pixels into
real-time steering, braking, and acceleration decisions — all on a
custom chip inside a consumer vehicle.

---

## 📚 Table of Contents

1. [Why Computer Vision?](#why-computer-vision)
2. [Camera System Architecture](#camera-system-architecture)
3. [HydraNet Neural Network](#hydranet-neural-network)
4. [Bird's Eye View & Occupancy Networks](#birds-eye-view--occupancy-networks)
5. [Traffic Light & Sign Recognition](#traffic-light--sign-recognition)
6. [Object Tracking Across Time](#object-tracking-across-time)
7. [Data Flywheel](#data-flywheel)
8. [Key Challenges](#key-challenges)
9. [Critical Evaluation](#critical-evaluation)
10. [References](#references)

---

## 🎯 Why Computer Vision?

Tesla's core argument: if humans drive using only their eyes,
a sufficiently powerful artificial vision system should be able
to do the same.

- Radar removed in **2021** — caused phantom braking from bridges
- LiDAR never adopted — too expensive, doesn't scale
- Result: forced the neural network to resolve all ambiguity
  using vision alone

---

## 📷 Camera System Architecture

**8 cameras, 360° coverage, each designed for a specific role:**

| Camera | Field of View | Primary Role |
|---|---|---|
| Front Wide-Angle | 120° | Nearby obstacles, intersections |
| Front Main | 45° | General driving, lane following |
| Front Telephoto | 25° | Long-range detection (100m+) |
| Left Pillar | 90° | Lane changes, merging |
| Right Pillar | 90° | Lane changes, merging |
| Rear-Left | 90° | Blind spot, overtaking vehicles |
| Rear-Right | 90° | Blind spot, overtaking vehicles |
| Rear | 120° | Reversing, parking |

**Why overlapping coverage?**
- Stereo-like depth perception from two cameras seeing same object
- Redundancy if one camera is obscured by rain or mud
- Seamless object tracking when moving between camera zones

---

## 🧠 HydraNet Neural Network

Tesla's multi-task neural network — one shared backbone, 50+ task heads.
```
8 Camera Feeds
      ↓
┌─────────────────────────┐
│   Shared CNN Backbone   │  ← RegNet architecture
│   (Feature Extraction)  │  ← Pays cost ONCE for all 50+ tasks
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│  Multi-Camera           │  ← Transformer attention across
│  Transformer Fusion     │    all 8 camera feature maps
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│  Temporal Fusion        │  ← ~0.5s of frame history (18 frames)
│  Transformer            │  ← Derives velocity & heading visually
└─────────────────────────┘
      ↓
┌──────────────────────────────────────────────────┐
│              50+ Task Heads (parallel)           │
│  Lane Detection · Object Detection · Depth       │
│  Traffic Lights · Sign OCR · BEV Map             │
│  Occupancy · Speed · Pedestrian Intent · ...     │
└──────────────────────────────────────────────────┘
      ↓
┌─────────────────────────┐
│   Planning Module       │  ← Speed · Steering · Lane choice
│   (End-to-end learned)  │  ← No manually written rules
└─────────────────────────┘
```

**Key advantage:** Gradients from all 50+ tasks train the shared
backbone simultaneously — lane detection makes object detection
better, and vice versa. Tasks teach each other.

---

## 🗺️ Bird's Eye View & Occupancy Networks

### Bird's Eye View (BEV)
- Transforms multi-camera features into a **top-down metric map**
- All objects represented in **real-world metres** from the car
- Makes path planning consistent regardless of which camera
  detected each object

### Occupancy Network
- Divides surrounding 3D space into **voxel grid**
- Each voxel classified as **free** or **occupied**
- Captures true shape of irregular objects (debris, fallen cargo)
- Explicitly models **driveable free space**
- Far more powerful than traditional bounding boxes

---

## 🚦 Traffic Light & Sign Recognition

### Traffic Light Pipeline
1. **Detection** — Telephoto camera localises signal bounding box
2. **State Classification** — Red / Amber / Green / Arrow / Flashing
3. **Temporal Smoothing** — Multiple frames must agree before acting
4. **Map Prior** — GPS confirms intersection type & expected signals

### Stop Sign Verification (3 checks must all agree)
- ✅ Octagonal shape
- ✅ Red colour (verified in HSV colour space)
- ✅ Text "STOP" via OCR

### Speed Limit Recognition
- Dedicated OCR task head reads digits from sign
- Cross-checked against map data for unit (mph vs km/h)
- System holds last confirmed speed limit in memory until new sign

---

## 📡 Object Tracking Across Time

Each tracked object carries a **state vector**:
`position · velocity · heading · bounding box dimensions`

**Kalman Filter — 2 steps per frame:**
- **Predict:** Where should this object be given its velocity?
- **Update:** New detection arrived — blend prediction with
  measurement weighted by confidence

**Key features:**
- Objects hidden behind trucks are **not deleted** — position
  continues to be predicted
- Tracking in **world coordinate space** (not pixel space) —
  no handoff problem when object moves between cameras
- Trajectory forecasts **3–5 seconds** into the future
- Time-to-collision computed → brakes earlier than human reaction

---

## 🔄 Data Flywheel

Tesla's biggest competitive advantage — **millions of vehicles
collecting training data continuously.**
```
Fleet Collection → Shadow Mode → Auto-Labelling → Dojo Training → OTA Deployment
      ↑                                                                   ↓
      └───────────────────── Smarter Model ──────────────────────────────┘
```

| Stage | What Happens |
|---|---|
| **Fleet Collection** | Edge cases trigger clip recording (near-misses, hard braking) |
| **Shadow Mode** | FSD runs silently, predicts actions, flags disagreements with driver |
| **Auto-Labelling** | Existing model labels clips; humans verify rare/novel cases |
| **Dojo Training** | Custom D1-chip supercomputer retrains on expanded dataset |
| **OTA Deployment** | Entire global fleet becomes smarter simultaneously |

---

## ⚠️ Key Technical Challenges

| Challenge | How Tesla Addresses It |
|---|---|
| Depth without LiDAR | Stereo geometry + temporal parallax + bounding box growth rate |
| Adverse weather | Fleet-collected diverse weather data; remains hardest problem |
| Long-tail scenarios | Shadow mode finds rare events at scale across millions of cars |
| Regional variation | Single model trained on international data from global fleet |

---

## ⚖️ Critical Evaluation

### ✅ Strengths
- Cheaper hardware — no LiDAR means lower manufacturing cost
- Scales to millions of vehicles at near-zero marginal cost
- Data flywheel competitors with smaller fleets cannot replicate
- End-to-end learned — no manual rules to maintain

### ❌ Limitations
- Camera depth estimation noisier than direct LiDAR measurement
- Single sensor modality = less inherent redundancy
- Severely degraded image quality forces reliance on map priors
- Low-visibility conditions harder for cameras than radar

---

## 📝 Conclusion

Tesla's FSD demonstrates how far camera-based computer vision can go
when paired with sufficient data, compute, and architectural innovation.
The HydraNet multi-task architecture, BEV world modelling, occupancy
networks, Kalman-filter tracking, and shadow mode data flywheel are
each significant achievements individually. Together they form a system
that has accumulated billions of autonomous miles and improves with
every software update.

Tesla has shown that camera-based perception, when scaled with
sufficient data and compute, is a serious contender for production
autonomous driving — and a masterclass in applied computer vision
engineering.

---

## 📄 Case Study Document

📎 [`Tesla_case_study_CV.docx`](./Tesla_case_study_CV.docx)

---

## 🔗 References

- Tesla AI Day 2021 — HydraNet and multi-task learning (Andrej Karpathy)
- Tesla AI Day 2022 — Occupancy networks, BEV, Dojo supercomputer
- Karpathy, A. (2021). "Building the Software 2.0 Stack." Tesla Engineering Blog.
- Radosavovic et al. (2020). "Designing Network Design Spaces." CVPR 2020. *(RegNet)*
- Geiger, A. et al. (2012). "Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite." CVPR 2012.
- Zhou, X. et al. (2019). "Objects as Points." arXiv:1904.07850.
- Tesla FSD Beta release notes — ongoing software changelog, tesla.com

---
** Priya A **
*PGDM Artificial Intelligence & Data Science | Computer Vision Course | Trimester 3*
