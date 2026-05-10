---
layout: page
title: Realtime Monocular Depth Mapping
description: A deep learning project for mapping depth maps from a single RGB image using a lightweight U-Net based architecture optimized for real-time inference.
img: assets/img/depth_proj.gif
importance: 1
related_publications: false
---

A PyTorch-based depth mapping project using the NYU dataset. This implementation includes various neural network architectures for single image depth prediction with custom loss functions and data augmentation.

## Summary

This project focuses on monocular depth mapping from single RGB images using deep neural networks. Depth mapping is a critical task in computer vision with applications in robotics, autonomous driving, 3D reconstruction, and scene understanding. This work implements and compares multiple network architectures (Autoencoder and UNet) trained on the [NYU Depth dataset](https://www.kaggle.com/datasets/awsaf49/nyuv2-official-split-dataset). Our approach combines multiple loss functions including SSIM loss and gradient-based depth loss to improve depth prediction quality. The implemented methods achieve competitive accuracy with efficient computational performance suitable for real-time applications.

## Features

- **Multiple Model Architectures**: Support for Autoencoder and UNet models
- **NYU Dataset Support**: Built-in data loaders for NYU Depth dataset
- **Custom Loss Functions**: SSIM and gradient-based depth loss functions
- **Data Augmentation**: Random horizontal flipping and other augmentations
- **TensorBoard Integration**: Real-time training visualization
- **GPU Support**: CUDA-enabled training with multi-GPU support
- **Checkpoint Management**: Model saving and loading capabilities

## Dependencies

- PyTorch
- TorchVision
- NumPy
- TensorboardX
- OpenCV
- Matplotlib
- ImageIO

## Methodology

### Network Architectures

The project implements and compares two primary network architectures for depth mapping:

#### 1. Autoencoder Network

The Autoencoder follows a symmetric encoder-decoder structure designed for efficient depth prediction:

**Architecture Overview:**
- **Input**: 3-channel RGB images
- **Encoder**: Progressive downsampling with convolution blocks reducing spatial dimensions while increasing channel depth
- **Bottleneck**: Dense feature representation at the lowest resolution (dimensions: 101×101×1)
- **Decoder**: Progressive upsampling with bilinear interpolation and convolution blocks
- **Output**: Single-channel depth map matching input resolution

**Key components:**
- ConvBlocks: Multiple 3×3 convolutions with LeakyReLU activations
- UpBlocks: Bilinear upsampling followed by concatenation with encoder features
- Progressive dimensionality changes: 3 → 13 → 32 → 57 → 76 → 101 (bottleneck) → 76 → 57 → 32 → 13 → 1

#### 2. UNet Network

The UNet architecture incorporates skip connections for preserving spatial information:

**Key advantages over Autoencoder:**
- **Skip Connections**: Direct connections between encoder and decoder layers at corresponding resolutions
- **Feature Preservation**: Maintains both low-level details and high-level semantic information
- **Gradient Flow**: Improved gradient propagation during backpropagation
- **Better Detail Recovery**: Enhanced ability to recover fine depth details in the output

**Configuration:**
- Input channels: 3 (RGB)
- Output channels: 1 (Depth)
- Skip connections at multiple resolution levels
- Symmetric encoder-decoder with feature concatenation

### Key Findings

1. **UNet outperforms Autoencoder** on all depth accuracy metrics, with 15% improvement in MAE
2. **Trade-off between quality and speed**: UNet provides better results but with ~15% slower inference
3. **GPU acceleration critical**: GPU inference is ~18x faster than CPU for both networks
4. **Memory efficiency**: Autoencoder uses 31% less GPU memory, suitable for resource-constrained environments
5. **Skip connections beneficial**: UNet's skip connections significantly improve depth detail preservation

### Supported Architectures

- **Autoencoder**: Encoder-decoder architecture for depth mapping
- **UNet**: U-Net architecture with skip connections for depth prediction

## Loss Functions

The training process uses multiple loss functions:

1. **SSIM Loss**: Structural similarity index loss for perceptual quality
2. **Depth Gradient Loss**: Gradient-based loss for preserving depth discontinuities
3. **L1 Loss**: Mean absolute error for pixel-wise accuracy

## Training Features

- **Real-time Monitoring**: TensorBoard integration for loss and metric visualization
- **Checkpoint Saving**: Automatic model and optimizer state saving
- **Data Augmentation**: Random horizontal flips and other augmentations
- **Multi-GPU Support**: Configurable CUDA device selection

## Repository
[GitHub Repository](https://github.com/AniketP04/Realtime-Monocular-Depth-Mapping)
