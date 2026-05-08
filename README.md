# Deep Compression of a Multimodal Fruit Classification Network

> Implementing Pruning, Trained Quantization, and Huffman Coding on a custom CNN–MLP architecture with LBP, Canny, and Handcrafted Features — End Eval Project, IIT Kanpur

---

## 🏆 Key Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Compression Ratio | ≥ 9× | **141.5×** |
| Accuracy Drop | ≤ 1.5% | **0.37%** |
| Bits per Weight | ≈ 3.57 | **1.0291** |
| Baseline Accuracy | — | **95.65%** |

---

## 📖 Overview

Deep Neural Networks achieve state-of-the-art performance but are expensive to deploy on resource-constrained devices (mobile, edge, IoT). This project faithfully implements the **Deep Compression pipeline of Han et al.** on a novel multimodal architecture trained for **257-class fruit classification** on the Fruits-360 dataset.

Unlike standard compressed models, our architecture does not use raw RGB images. Instead, it extracts:
- **Texture information** via Local Binary Patterns (LBP)
- **Structural information** via Canny edge maps
- **12 handcrafted** color and shape features

These are fused through a dual-branch CNN + MLP architecture and then compressed through a 3-stage pipeline.

---

## 📁 Project Structure

```
DNN_Compression/
├── compression/
│   ├── conv2d.py          # Compression-aware Conv2d with pruning mask
│   ├── linear.py          # Compression-aware Linear with pruning mask
│   ├── prune.py           # Model-wide magnitude pruning
│   └── quantization.py    # K-Means quantization across layers
├── data/
│   └── data_loader.py     # OpenCV preprocessing, LBP, Canny, HSV, contour features
├── models/
│   └── model_cifar.py     # SmallCIFARNet dual-branch hybrid CNN
├── utils/
│   ├── training.py        # Adam training loop with per-epoch logging
│   ├── test_eval.py       # Evaluation and per-class accuracy
│   └── loading.py         # save/load model as sparse .npz
└── main.py                # End-to-end pipeline driver
```

---

## Model Architecture — SmallCIFARNet

### 1. Dual CNN Branches
Two independent CNN branches (one for LBP, one for Canny), each processing a `1×100×100` input:
Block 1: Conv2d(1→16)  + ReLU + MaxPool → 16×50×50
Block 2: Conv2d(16→32) + ReLU + MaxPool → 32×25×25
Block 3: Conv2d(32→64) + ReLU + MaxPool → 64×12×12
Flatten → 9,216-d vector

### 2. Handcrafted Feature Encoder
A 2-layer MLP lifts the 12-dimensional handcrafted feature vector to 128 dimensions before fusion.

### 3. Fusion Classifier
[LBP (9216) ‖ Canny (9216) ‖ Handcrafted (128)] → 18,560-d
→ Linear(18560 → 1024) + ReLU + Dropout(0.4)
→ Linear(1024 → 512)   + ReLU + Dropout(0.3)
→ Linear(512 → 256)    + ReLU
→ Linear(256 → 257)    [logits]

| Component | Parameters | Share |
|-----------|-----------|-------|
| LBP CNN Branch | 9,788,544 | 49.5% |
| Canny CNN Branch | 9,788,544 | 49.5% |
| Feature Encoder | 9,088 | 0.05% |
| MLP Classifier | 198,209 | 1.0% |
| **Total** | **19,784,385** | 100% |

---

## 🔧 Feature Engineering

### Local Binary Pattern (LBP)
Encodes local texture by comparing each pixel to its 8 neighbors. Vectorized with NumPy for ~100× speedup over naive implementations.

### Canny Edge Map
Binary edge map using thresholds `[100, 200]`, capturing fruit silhouettes and boundaries.

### Handcrafted Features (12-dim)
- **Color (6-dim):** Mean and std of H, S, V channels in HSV space
- **Shape (6-dim):** Area ratio, aspect ratio, solidity, circularity, and 2 log Hu moments

---

##  Deep Compression Pipeline
Baseline Training → Pruning → Retrain → Quantization → Retrain → Huffman Coding → Deploy
10 epochs       τ=0.05    5 epochs    k=16           5 epochs    1.03 bits/w    0.33 MB

### Stage 1 — Baseline Training

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam |
| Learning Rate | 1e-3 |
| LR Scheduler | Cosine Annealing (η_min = 1e-5) |
| Loss | Cross-Entropy + Label Smoothing (ε=0.05) |
| Gradient Clipping | ‖∇‖₂ ≤ 1.0 |
| Batch Size | 64 |
| Epochs | 10 |

### Stage 2 — Magnitude-Based Pruning
Weights with `|W_ij| < τ = 0.05` are zeroed and masked permanently via registered buffer masks.

- **99.7% of weights pruned** (19,715,118 / 19,781,920)
- Accuracy after pruning (before retrain): **40.79%**
- Accuracy after 5-epoch recovery: **95.24%**

### Stage 3 — K-Means Quantization
Non-pruned weights clustered into `k=16` centroids (4-bit effective precision), fine-tuned for 5 epochs.

- Accuracy after quantization (before retrain): **92.44%**
- Accuracy after 5-epoch recovery: **95.29%**
- Effective bits/weight: **0.0135**

### Stage 4 — Huffman Entropy Coding
Min-heap Huffman tree built per layer. With 99.7% sparsity, zero-centroid dominates → ~1 bit per weight.

- **Average bits/weight: 1.0291**
- **Gain over quantized: 15.55×**
- **Final size: 0.33 MB** (down from 47.30 MB)

---

## 📊 Results

### Accuracy at Each Stage

| Stage | Accuracy | Δ vs Baseline |
|-------|----------|---------------|
| Baseline | 95.65% | — |
| After Pruning (before retrain) | 40.79% | −54.86% |
| After Pruning (recovered) | 95.24% | −0.41% |
| After Quantization (before retrain) | 92.44% | −2.80% |
| After Quantization (recovered) | 95.29% | −0.37% |
| **Final (Huffman)** | **95.29%** | **−0.37%** |

### Storage at Each Stage

| Model | Size (MB) | Ratio |
|-------|-----------|-------|
| Baseline | 47.30 | 1× |
| After Pruning | 4.83 | 9.8× |
| After Quantization | 4.83 | 9.8× |
| **Huffman Compressed** | **0.33** | **141.5×** |

### vs. Original Han et al. Paper

| Metric | Paper Target | Han (AlexNet) | Ours |
|--------|-------------|----------------|------|
| Compression Ratio | ≥ 9× | 9× | **141.5×** |
| Accuracy Drop | ≤ 1.5% | < 1% | **0.37%** |
| Bits per Weight | ≈ 3.57 | 3.57 | **1.0291** |
| Sparsity | ≈ 90% | 89% | **99.7%** |

---

##  Compression-Aware Layer Design

`modified_conv2d` and `modified_linear` subclass PyTorch's standard layers and add:

- **Pruning mask** `M ∈ {0,1}` stored as `register_buffer` (persists in `state_dict`)
- **Mode flag** (0=normal, 1=prune, 2=quantize) also stored as a buffer
- **Masked forward pass:** `y = f(W ⊙ M, x, b)` — pruned weights never contribute even during retraining

---

##  Setup & Usage

### Requirements
```bash
pip install torch torchvision opencv-python scikit-learn numpy
```

### Dataset
Download [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) from Kaggle and place it in the project root.

### Run
```bash
python main.py
```

---

##  Hardware

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 7435HS |
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GB VRAM) |
| RAM | 16 GB |
| Framework | PyTorch 2.x, OpenCV, scikit-learn, NumPy |
| Python | 3.10.0 |


---

##  Reference

Han, S., Mao, H., & Dally, W. J. (2015). [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149). *ICLR 2016*.
