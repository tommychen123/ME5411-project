# Character Recognition Pipeline (MATLAB)

This repository implements a **complete character recognition pipeline** for grayscale or binary images,
including preprocessing, segmentation, and classification via **CNN** and **MLP**.

---

## 📂 Directory Structure

```
src/
│
├── main.m                     # Entry point to run all steps end-to-end
├── step1_display.m            # Image loading and visualization
├── step1_enhancement.m        # Image enhancement (CLAHE, normalization, etc.)
├── step2_filter.m             # Noise filtering
├── step3_roi.m                # ROI extraction
├── step4_binarize.m           # Image binarization
├── step5_outline.m            # Outline extraction and morphology
├── step6_segment.m            # Character segmentation (outputs cropsBin)
│
├── step7_dataset.m            # Dataset preparation for CNN/MLP
├── step7_cnn.m                # CNN model training
├── step7_task1_cnn.m          # Improved CNN model definition & training
├── step7_task1_apply_cnn.m    # CNN inference (auto retry, MLP assistance for low confidence)
│
├── step7_task2_mlp.m          # MLP training (fully connected classifier)
├── step7_task2_apply_mlp.m    # MLP inference (command-line output + grid visualization)
│
└── tools/                     # Utility functions (e.g. augmentation, helpers)
```

---

## 🧠 Features

- **End-to-End Workflow:** From raw image → segmentation → classification  
- **Two Classification Branches:**  
  - CNN: Convolutional Neural Network  
  - MLP: Multi-layer Perceptron (non-CNN baseline)  
- **Unified Preprocessing:**  
  White background, black characters, centered with controllable padding (`padScale`)  
- **Confidence-aware Fusion:**  
  If CNN confidence < 0.7, system automatically retries or invokes MLP for verification  
- **Visualization:**  
  - `step7_task1_cnn_apply`: `step7_task1_cnn_inputs_grid.png`  
  - `step7_task2_apply_mlp`: `step7_task2_mlp_inputs_grid.png`  

---

## 🧩 Requirements

- MATLAB R2021a or later
- (Optional) GPU support for faster CNN training

---

## 🗂 Dataset Structure

Each subfolder represents one character class:

```
../data/dataset_2025/
│
├── 0/       # Digits
├── 4/
├── 7/
├── 8/
├── A/       # Letters
├── D/
└── H/
```

> Use consistent file naming (e.g. `img001_0001.png`), and ensure all are 128×128 grayscale or binary images.

---

## 🚀 Quick Start

### A. Full Pipeline
```matlab
main
```

### B. CNN Training
```matlab
step7_cnn
```

### C. CNN Inference
```matlab
state = step7_task1_apply_cnn(state, cfg);
```

### D. MLP Training
```matlab
step7_task2_mlp
```

### E. MLP Inference
```matlab
state = step7_task2_apply_mlp(state, cfg);
```

---

## 📁 Output Structure

```
../results/
├── models/
│   ├── CNN_latest.mat
│   └── MLP_latest.mat
│
├── figures/

```

---

## ⚙️ Key Parameters

| Parameter | Description | Typical |
|------------|-------------|----------|
| `padScale` | Controls padding (white border) | 1.4–1.7 |
| `useCLAHE` | Contrast enhancement | true |
| `lowConfThr` | Confidence threshold for retry/MLP assist | 0.7 |
| `retryPadScale` | Secondary smaller pad for retry | 1.2 |

---

## 🧩 Fusion Logic (Simplified)

```
For each segmented character:
    Run CNN → get (label, confidence)
    if confidence < 0.7:
        Retry with smaller padScale
        if still low:
            use MLP → verify
Output final label (from best result)
```


