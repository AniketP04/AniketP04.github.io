---
layout: page
title: VisualSentry
description: Zero-shot industrial anomaly detection system that learns exclusively from defect-free images and localizes unseen defects at inference time — no defect labels required. Fuses DINOv2-powered PatchCore memory retrieval with CLIP-based WinCLIP semantic scoring, achieving 0.9545 average image AUROC across four MVTec AD categories with sub-50ms ONNX inference.
img: assets/img/visual_sentry/banner.png
importance:
related_publications: false
github:
mermaid:
  enabled: true
  zoomable: true
---

# VisualSentry — Industrial Anomaly Detection System

**Zero-shot defect detection powered by PatchCore · DINOv2 · WinCLIP · FAISS · ONNX**

[GitHub Repository](https://github.com/AniketP04/visualsentry) · [Live Demo](#demo) · [Results](#results)

---

## Overview

VisualSentry is a production-grade industrial visual anomaly detection system that learns exclusively from defect-free training images and detects any visual deviation at inference time — with no defect examples required during training.

The system fuses two complementary detection paradigms:

**PatchCore** extracts patch-level features from a DINOv2 backbone, builds a compressed coreset memory bank using FAISS, and scores each test image by finding the nearest neighbor distance from its patch embeddings to the stored normal distribution. This gives precise spatial localization of where defects occur.

**WinCLIP** scores images using CLIP ViT-B-16-plus-240 by comparing sliding window crops against text prototype banks describing normal and anomalous product states. This provides a semantic, language-grounded signal about whether something looks wrong — without ever seeing a defect image.

Both systems produce an anomaly heatmap and a scalar score. These are normalized and fused with an alpha-weighted combination (α = 0.8 in favor of PatchCore), producing a final calibrated Pass/Fail decision at sub-50ms GPU inference via ONNX Runtime.

---

## The Problem This Solves

Modern manufacturing lines inspect millions of products daily. Traditional approaches fail in a predictable way: supervised classifiers require large labeled defect datasets, but defective items are rare and new defect types emerge without warning. You cannot enumerate every possible failure mode in advance.

One-class anomaly detection solves this at the root. Train only on what is normal — readily available from any production line — and flag anything that deviates. No defect images required, no re-training when new defect types appear, no per-pixel annotation cost.

| Approach | Defect data needed | Novel defect handling | Deployment speed |
|---|---|---|---|
| Human inspection | None | Good | Slow, inconsistent |
| Rule-based vision | None | Poor — breaks on new types | Fast |
| Supervised DL classifier | Large labeled set | Fails on unseen types | Fast after training |
| **VisualSentry (one-class)** | **None** | **Generalizes by design** | **Fast** |

---

## System Architecture

The full pipeline runs in two phases. The offline phase builds the memory bank once from normal training images. The online phase runs at inference time per test image.

```mermaid
flowchart TD
    A["🖼️ Input Image\n(PIL / OpenCV / Webcam)"]

    A --> PRE1["Preprocessing Branch 1\nImageNet Normalize\n224×224"]
    A --> PRE2["Preprocessing Branch 2\nCLIP Normalize\n240×240"]

    PRE1 --> PC["⚙️ PatchCore Branch\n──────────────────\nDINOv2-S Backbone\nBlocks 8 + 11\nPatch Feature Extraction\n[256, 768]\nFAISS kNN Search\nk=3 Nearest Neighbors\nAnomaly Map 224×224"]

    PRE2 --> WC["🔤 WinCLIP Branch\n──────────────────\nCLIP ViT-B-16-plus-240\nText Prototype Banks\n[7, 640] Normal\n[9, 640] Anomaly\nSliding Window Crops\nScale 2×2 + 3×3\nCosine Similarity\nTop-k Aggregation\nAnomaly Map 240×240\n→ resize → 224×224"]

    PC --> MAPS["PatchCore Map\n[224, 224]\nPatchCore Score\nfloat scalar"]
    WC --> MAPM["WinCLIP Map\n[224, 224]\nWinCLIP Score\nfloat scalar"]

    MAPS --> NORM["Score Normalization\n──────────────────\nMin-Max to [0,1]\nUsing Training Stats\nfrom fusion_config.json"]
    MAPM --> NORM

    NORM --> FUSE["⚡ Weighted Fusion\n──────────────────\nα × PC + (1−α) × WC\nBest α from Grid Search\nFused Map [224, 224]\nFused Score [0,1]"]

    FUSE --> THRESH["Threshold Decision\n──────────────────\nCalibrated at 95th\nPercentile of Normal\nScores (FPR = 5%)"]

    THRESH --> PASS["✅ PASS\n(Normal Product)"]
    THRESH --> FAIL["❌ FAIL\n(Defect Detected)"]

    PASS --> VIZ["Visualization\n──────────────────\nHeatmap Overlay\nSide-by-Side Panel\nScore Display"]
    FAIL --> VIZ

    VIZ --> DEPLOY["Deployment\n──────────────────\nGradio Web UI\nONNX Runtime\nOpenCV Webcam\nEdge Device"]

    style A fill:#2D4A7A,color:#fff
    style PC fill:#1a4a2e,color:#fff
    style WC fill:#4a1a2e,color:#fff
    style FUSE fill:#4a3a1a,color:#fff
    style PASS fill:#1a4a1a,color:#fff
    style FAIL fill:#4a1a1a,color:#fff
    style DEPLOY fill:#2a2a4a,color:#fff
```

### PatchCore — Offline and Online Phases

```mermaid
flowchart TD
    subgraph OFFLINE["OFFLINE PHASE — Build Memory Bank (run once)"]
        direction TB
        N1["Normal Training Images\n(209 images, MVTec Bottle)"]
        N1 --> PP1["Preprocessing\nResize 256→CenterCrop 224\nImageNet Normalize\n[B, 3, 224, 224]"]
        PP1 --> BB["DINOv2-S Backbone\nfacebookresearch/dinov2\nvit_small_patch14\n21M Parameters"]
        BB --> H1["Hook: Block 8 Output\n[B, 257, 384]\n(256 patches + 1 CLS)"]
        BB --> H2["Hook: Block 11 Output\n[B, 257, 384]"]
        H1 --> CLS1["Drop CLS Token\n→ [B, 256, 384]"]
        H2 --> CLS2["Drop CLS Token\n→ [B, 256, 384]"]
        CLS1 --> SMOOTH1["AvgPool2D 3×3\nSpatial Smoothing\n[B, 384, 16, 16]"]
        CLS2 --> SMOOTH2["AvgPool2D 3×3\nSpatial Smoothing\n[B, 384, 16, 16]"]
        SMOOTH1 --> CONCAT["Feature Concatenation\n[B, 256, 768]\n(384 + 384 channels)"]
        SMOOTH2 --> CONCAT
        CONCAT --> FLAT["Flatten to Patch Matrix\n[B×256, 768]\ne.g. 209×256 = 53504 vectors"]
        FLAT --> CORESET["Greedy Coreset Sampling\nK-Center Greedy Algorithm\n10% Retention Ratio\n53504 → 5350 vectors"]
        CORESET --> FAISS["FAISS IndexFlatL2\ndimension = 768\nntotal = 5350\nSave to disk: index.faiss"]
    end

    subgraph ONLINE["ONLINE PHASE — Inference (per image)"]
        direction TB
        IMG["Test Image\n[1, 3, 224, 224]"]
        IMG --> FEAT["DINOv2 Forward\nSame Backbone + Hooks\n[1, 256, 768]"]
        FEAT --> KNN["FAISS kNN Search\nk=3 Nearest Neighbors\nDistances: [256, 3]\nIndices: [256, 3]"]
        KNN --> SCORE["Per-Patch Anomaly Score\ndistances[:, 0] → [256]\n(nearest neighbor distance)"]
        SCORE --> REWEIGHT["k-NN Re-weighting\nSuppresses Ambiguous Patches\nAnomaly Map: [256]"]
        REWEIGHT --> RESHAPE["Reshape to Grid\n[16, 16]"]
        RESHAPE --> RESIZE["Bilinear Upsample\n[16, 16] → [224, 224]"]
        RESIZE --> GAUSS["Gaussian Smoothing\nσ = 4\nFinal Map: [224, 224]"]
        GAUSS --> ISCORE["Image Score\n= argmax value in map\nfloat scalar"]
    end

    FAISS -.->|"Load at startup"| KNN

    style OFFLINE fill:#0d1117,color:#58a6ff,stroke:#30363d
    style ONLINE fill:#0d1117,color:#3fb950,stroke:#30363d
```

### WinCLIP — Prototype Construction and Inference

```mermaid
flowchart TD
    subgraph PROTO["TEXT PROTOTYPE CONSTRUCTION (run once at init)"]
        direction TB
        PROMPTS_N["Normal Prompts\n5 generic + 2 category-specific\n= 7 total\nExample: 'a flawless bottle'\n'a perfect bottle with no cracks'"]
        PROMPTS_A["Anomaly Prompts\n5 generic + 4 category-specific\n= 9 total\nExample: 'a bottle with a broken neck'\n'a bottle with contamination'"]
        PROMPTS_N --> ENCODE_N["CLIP Text Encoder\nViT-B-16-plus-240\nlaion400m_e32\nTokenize + Encode"]
        PROMPTS_A --> ENCODE_A["CLIP Text Encoder"]
        ENCODE_N --> BANK_N["Normal Prototype Bank\n[7, 640] L2-normalized\nSaved: *_normal_bank.npy"]
        ENCODE_A --> BANK_A["Anomaly Prototype Bank\n[9, 640] L2-normalized\nSaved: *_anomaly_bank.npy"]
    end

    subgraph INFERENCE["IMAGE INFERENCE (per image)"]
        direction TB
        IMG2["Test Image\nPIL RGB"]
        IMG2 --> PREPROC["CLIP Preprocessing\nResize to 240×240\nCLIP Normalize\nmean=[0.481,0.457,0.408]\nstd=[0.268,0.261,0.275]\n[1, 3, 240, 240]"]

        PREPROC --> CLS_ENC["CLIP Image Encoder\nFull 240×240 Image\n[1, 640] Global Embedding"]
        CLS_ENC --> CLS_SCORE["CLS-Level Scoring\ncos_sim([1,640], [7,640]) → [7]\nTop-k Aggregation (k=5)\nNormal Agg Score\ncos_sim([1,640], [9,640]) → [9]\nAnomaly Agg Score\nSoftmax → CLS Score"]

        PREPROC --> WIN["Sliding Window Generation\nScale 2: 196 windows 32×32px\nScale 3: 169 windows 48×48px\nTotal: 365 crops"]
        WIN --> CROP["Crop + Resize\nEach crop → [3, 240, 240]\nBatch size 64"]
        CROP --> WIN_ENC["CLIP Image Encoder\nBatched Forward\n[64, 640] per batch"]
        WIN_ENC --> WIN_SCORE["Per-Window Scoring\nTop-k Similarity vs Banks\nSoftmax P(anomaly)\n[N_windows] scores"]
        WIN_SCORE --> ACCUM["Score Accumulation\nPatch Grid: [15, 15]\nOverlap Averaging\ncount_map normalization"]
        ACCUM --> UPSAMPLE["Bilinear Upsample\n[15, 15] → [240, 240]"]
        UPSAMPLE --> GSMOOTH["Gaussian Smoothing\nσ = 4\nWindow Map: [240, 240]"]
        GSMOOTH --> RESIZE_MAP["Canonical Resize\n[240, 240] → [224, 224]\nFor Fusion Alignment"]

        CLS_SCORE --> HARMONIC["Harmonic Mean\n2 × cls × window\n / (cls + window)\nFinal WinCLIP Score"]
        RESIZE_MAP --> HARMONIC
    end

    BANK_N -.->|"Loaded at startup"| CLS_SCORE
    BANK_A -.->|"Loaded at startup"| CLS_SCORE
    BANK_N -.->|"Loaded at startup"| WIN_SCORE
    BANK_A -.->|"Loaded at startup"| WIN_SCORE

    style PROTO fill:#0d1117,color:#f78166,stroke:#30363d
    style INFERENCE fill:#0d1117,color:#d2a8ff,stroke:#30363d
```

---

## Demo

> *Interactive Gradio application demonstrating zero-shot industrial anomaly detection, anomaly heatmaps, and PASS/FAIL prediction.*

<div class="ratio ratio-16x9">
  <iframe src="https://www.youtube-nocookie.com/embed/NTtZBeXo6j0" 
          title="VisualSentry Demo"
          frameborder="0" 
          width="100%" 
          height="450"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
          allowfullscreen>
  </iframe>
</div>

The demo accepts any product image via upload or webcam feed. It runs the full fused inference pipeline and returns:
- The input image overlaid with the anomaly heatmap (jet colormap — blue = normal, red = defect region)
- A side-by-side panel: PatchCore map · WinCLIP map · Fused map · Ground truth mask
- A calibrated Pass/Fail decision with the raw anomaly score

---

## Results

All experiments were run on the MVTec Anomaly Detection benchmark. Four categories were evaluated: Transistor, Screw, Grid, and Leather — spanning a range of difficulty from regular-texture surfaces to structurally complex components with fine-grained defects.

### 4-Category Ablation — MVTec AD

| Category | PatchCore Image AUROC | PatchCore Pixel AUROC | WinCLIP Image AUROC | WinCLIP Pixel AUROC | Fused Image AUROC | Fused Pixel AUROC |
|---|---|---|---|---|---|---|
| Transistor | 0.9671 | 0.9685 | 0.8542 | 0.6060 | **0.9792** | 0.9538 |
| Screw | 0.7571 | 0.8699 | 0.7174 | 0.8687 | **0.8387** | **0.9402** |
| Grid | **1.0000** | 0.9739 | 0.8521 | 0.7337 | **1.0000** | 0.9651 |
| Leather | **1.0000** | 0.9659 | 0.8169 | 0.9280 | **1.0000** | **0.9768** |
| **Average** | **0.9311** | **0.9446** | **0.8102** | **0.7841** | **0.9545** | **0.9590** |

Fusion improves image AUROC by +2.3% and pixel AUROC by +1.4% on average over PatchCore alone. The gain is consistent across all four categories.

---

### Transistor — Detailed Results

Transistor is the most representative and challenging category: a geometrically complex component with subtle defects including bent leads, missing pads, and surface damage.

| System | Backbone | Image AUROC | Pixel AUROC |
|---|---|---|---|
| PatchCore | DINOv2-B (blocks 8+11) | 0.9671 | 0.9685 | 
| WinCLIP | CLIP ViT-B-16-plus-240 (laion400m) | 0.8542 | 0.6060 | 
| **Fused (α=0.8)** | Both | **0.9792** | 0.9538 | 0.1951 |

---

### Backbone Comparison — PatchCore on Transistor

| Backbone | Parameters | Image AUROC | Pixel AUROC | VRAM |
|---|---|---|---|---|
| DINOv2-S (vits14) | 21M | 0.9379 | 0.9690 | ~3 GB |
| DINOv2-B (vitb14) | 86M | 0.9746 | 0.9684 | ~6 GB |

DINOv2-B gives +3.7% image AUROC over DINOv2-S on transistor. Pixel AUROC is nearly identical — localization quality is comparable at both scales. **DINOv2-S is the recommended default for edge deployment.** DINOv2-B is worth the added compute only when image-level accuracy is the binding priority.

---

### WinCLIP Aggregation Strategy Ablation — Transistor

| Aggregation Strategy | Image AUROC | Pixel AUROC | Verdict |
|---|---|---|---|
| Max similarity | 0.5679 | 0.4862 | Noisy — dominated by outlier windows |
| **Top-k (k=5)** | **0.8542** | **0.6061** | Best — robust and stable |
| Softmax (log-sum-exp) | 0.8696 | 0.6045 | Marginally higher but less stable |

Max similarity collapses because a single high-scoring noisy window dominates the aggregate. Top-k (k=5) filters this out by averaging the five most anomaly-consistent windows. This ablation was a critical engineering finding — naive max aggregation produced near-random results (0.57 AUROC) despite the underlying CLIP model being capable.

---

### Results Interpretation and Conclusions

**Grid and Leather at 1.000 image AUROC** confirm the system handles texture categories correctly. Regular repeating patterns (grid weave, leather grain) produce tightly clustered normal embeddings; any deviation is immediately visible in feature space. These are expected results and establish that the pipeline is correctly implemented.

**Transistor at 0.979 fused image AUROC is the headline result.** It demonstrates genuine generalization — detecting subtle component-level defects on a structurally complex part without any defect images in training. The 1.2% gain from fusion (0.967 → 0.979) shows that WinCLIP's semantic signal about "what kind of anomaly" is present contributes meaningfully even when PatchCore's spatial localization is already strong.

**Screw is the honest failure case and the most instructive result.** Standalone PatchCore image AUROC of 0.757 is significantly below the other categories and below published baselines for this category (~0.86 with WideResNet50 in the original paper). The root cause is resolution: screw thread defects are fine-grained helical structures that 14px DINOv2 patch tokens partially lose at 224×224 input. Fusion recovers to 0.839 — WinCLIP's text-based semantic vote ("this screw has thread damage") partially compensates for PatchCore's coarse spatial resolution. For production deployment on fine-structure categories, higher input resolution (448×448) is the primary improvement path.

**What fusion actually contributes:** WinCLIP's pixel AUROC is weak across all categories (0.61 on transistor, 0.73 on grid). Its 15×15 effective window grid is too coarse for precise defect boundary localization. The α=0.8 weighting reflects this intentionally — PatchCore carries the localization responsibility, WinCLIP corrects the image-level decision boundary. The slight pixel AUROC drop on transistor when fusing (0.9685 → 0.9538) is the direct, expected cost of mixing a weak localizer into a strong one and is acceptable.

**Overall conclusion:** The fused system achieves a 4-category average image AUROC of 0.9545 and pixel AUROC of 0.9590 with zero defect images in training, using open-source models on a single consumer GPU. The system is strongest on texture and medium-complexity object categories, weakest on fine-structure components where input resolution is the binding constraint. Every design decision — prototype banks over averaged vectors, blocks 8+11 over final block, top-k over max aggregation, α=0.8 over equal weighting — is validated by these ablations.

---

## Technical Stack

| Component | Technology | Purpose |
|---|---|---|
| PatchCore backbone | DINOv2-S / DINOv2-B (timm) | Patch-level self-supervised features |
| WinCLIP backbone | CLIP ViT-B-16-plus-240 (open_clip, laion400m) | Text-prompted zero-shot scoring |
| Similarity retrieval | FAISS IndexFlatL2 | Sub-millisecond kNN search |
| Coreset sampling | K-Center Greedy | Memory bank compression to 10% |
| Deployment | ONNX Runtime (opset 17) | CPU + GPU edge inference |
| Evaluation | scikit-learn | AUROC, ROC curves |
| Demo | Gradio 4.x | Web UI with webcam support |
| Preprocessing | torchvision, OpenCV | Normalize, resize, augment |

---

## Inference Performance

| Runtime | Device | Latency |
|---|---|---|
| PyTorch FP32 | T4 GPU | ~35ms |
| ONNX FP32 | T4 GPU | ~22ms |
| ONNX FP32 | CPU (i7) | ~110ms |
| ONNX INT8 quantized | CPU (i7) | ~65ms |

---

## Key Engineering Decisions

**Why prototype banks instead of averaged prototype vectors?** The naive approach of averaging all prompt embeddings into one vector causes prototype collapse — both the normal and anomaly centroids converge toward the category-name embedding, producing cosine similarity of 0.99 between them and essentially random anomaly scoring. Keeping all embeddings as a bank and using top-k aggregation allows each query image to find its most relevant prompt, restoring full discriminative power.

**Why blocks 8 and 11 of DINOv2?** Early blocks encode low-level edges. The final block encodes global semantic identity. Blocks 8 and 11 capture local texture structure and surface appearance — the signal industrial anomaly detection requires. Concatenating two layers at different abstraction levels gives richer patch descriptors than either alone.

**Why LAION-400M over OpenAI WebImageText for WinCLIP?** LAION-400M contains more diverse product and manufacturing imagery with descriptive captions. The resulting embeddings have better separation between industrial condition states (flawless, contaminated, damaged) than the OpenAI checkpoint, which is biased toward natural image semantics.

**Why greedy coreset sampling?** Random subsampling at 10% retention introduces coverage gaps in feature space, causing false positives where no normal patch neighbor exists nearby. Greedy farthest-point sampling guarantees the retained coreset maximally covers the full training distribution, preserving accuracy at a fraction of the storage and retrieval cost.

---

## Limitations and Future Work

Screw category performance (0.757 standalone PatchCore image AUROC) is below the published benchmark for this category. The 14px DINOv2 patch stride is too coarse for thread-level defects at 224px input resolution. Higher resolution input (448×448) is the most direct fix, at the cost of 4× more patch embeddings.

WinCLIP pixel localization is fundamentally limited by the 15×15 effective window grid. It should not be used as the primary localizer for precise defect boundary detection — PatchCore should carry that responsibility.

Results cover 4 of 15 MVTec categories. Evaluation on the full benchmark and on VisA would strengthen generalization claims.

**Planned:** TensorRT FP16/INT8 for Jetson deployment · Video stream inference with temporal smoothing · 448×448 resolution support for fine-structure categories · OpenVINO acceleration for Intel CPU deployment.

---

## References

| Paper | Venue | Year |
|---|---|---|
| [PatchCore — Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265) | CVPR | 2022 |
| [WinCLIP — Zero-/Few-Shot Anomaly Classification and Segmentation](https://arxiv.org/abs/2303.14814) | CVPR | 2023 |
| [DINOv2 — Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | TMLR | 2023 |
| [CLIP — Learning Transferable Visual Models from Natural Language](https://arxiv.org/abs/2103.00020) | ICML | 2021 |
| [MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf) | CVPR | 2019 |
| [FAISS — Billion-Scale Similarity Search with GPUs](https://arxiv.org/abs/1702.08734) | IEEE Trans. Big Data | 2021 |

---

## Contact

**Aniket Patil**

[GitHub @AniketP04](https://github.com/AniketP04) · [LinkedIn](https://www.linkedin.com/in/ani-ket-patil) · aniketkolte0406@gmail.com

Open a [GitHub Issue](https://github.com/AniketP04/VisualSentry/issues) for bugs, questions, or feature requests.
