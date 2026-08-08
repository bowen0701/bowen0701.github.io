---
author: Bowen Lee
date: '2026-08-07'
layout: post
tags:
- computer-vision
- convolution-neural-networks
- deep-learning
title: Conv2d - PyTorch 2D Convolutional Layer
---
## Table of Contents
{:.no_toc}

* TOC
{:toc}


`torch.nn.Conv2d` applies a 2D convolution over an input signal (e.g., an image).

## `Conv2d` API Signature

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
```

| Parameter | Meaning |
|---|---|
| `in_channels` | Number of channels in the input (e.g., 1 for grayscale, 3 for RGB) |
| `out_channels` | Number of filters (each produces one output channel) |
| `kernel_size` | Size of the sliding window (e.g., `3` means 3x3) |
| `stride` | How many pixels the kernel moves per step (default 1) |
| `padding` | Zeros added around the input border (default 0) |
| `bias` | Whether to add a learnable bias to each output channel (default True) |

## Example: `Conv2d(1, 32, 3, stride=2, padding=1)`

- **1** input channel (grayscale image)
- **32** output channels (32 learned 3x3 filters)
- **3x3** kernel: each filter is a 3x3 sliding window that computes a weighted sum over a 3x3 patch
- **stride=2**: the kernel jumps 2 pixels per step, halving spatial dimensions
- **padding=1**: 1 pixel of zeros added on each side, preventing excessive shrinkage

## Output Size Formula

For a square input of size $H$:

```
H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1
```

The same formula applies independently to width.

### Derivation

The formula counts how many times the kernel fits as it slides across the padded input:

1. **Pad the input**: effective size = `H_in + 2*padding`
2. **First kernel placement uses up `kernel_size` pixels**: remaining pixels to slide over = `H_in + 2*padding - kernel_size`
3. **Each step moves `stride` pixels**: number of additional placements = `floor(remaining / stride)`
4. **+1 for the first placement**: total = `floor((H_in + 2*padding - kernel_size) / stride) + 1`

### Worked example

Input `1x28x28` through `Conv2d(1, 32, 3, stride=2, padding=1)`:

```
H_out = floor((28 + 2*1 - 3) / 2) + 1
      = floor(27 / 2) + 1
      = 13 + 1
      = 14
```

Output shape: `32x14x14` (32 channels, each 14x14).

### Common cases

| Configuration                   | Effect on spatial size                    |
| ------------------------------- | ----------------------------------------- |
| `kernel=3, stride=1, padding=1` | Same size (preserves dimensions)          |
| `kernel=3, stride=2, padding=1` | Halves size (downsample)                  |
| `kernel=3, stride=1, padding=0` | Shrinks by 2 (loses border pixels)        |
| `kernel=1, stride=1, padding=0` | Same size (1x1 conv, mixes channels only) |

**What does 1x1 conv mean?**
It's essentially a per-pixel fully connected layer. Common uses:
- Channel reduction: e.g., 256 channels → 64 channels (cheaper than 3x3 conv on 256 channels)
- Adding nonlinearity: 1x1 conv + ReLU lets the network learn cross-channel interactions without changing spatial size

## What Each Filter Learns

Each filter is a 3D block of learnable weights shaped `in_channels x kernel_size x kernel_size`. It must match the input's channel depth so it can compute a dot product across **all channels simultaneously** at each spatial position.

For `Conv2d(3, 32, 3)` (RGB input, 32 filters, 3x3 kernel):

```
One filter shape: 3 x 3 x 3  (in_channels x kernel_h x kernel_w)
                  = 27 weights

It slides across the input and at each position:
  - looks at a 3x3 patch across ALL 3 RGB channels
  - computes: sum of (27 weights * 27 input values) + bias
  - produces: 1 scalar

One filter sliding across all positions → 1 output channel (a 2D feature map)
32 filters → 32 output channels
```

A single filter spans the **full channel depth** of the input. It doesn't convolve over one channel at a time. This is how Conv2d learns cross-channel patterns (e.g., a filter that detects edges where red meets green).

Early layers learn low-level features (edges, textures); deeper layers learn higher-level patterns.

The total number of learnable parameters: `out_channels * (in_channels * kernel_size^2 + 1)` (the +1 is the bias per filter, if `bias=True`).

## References

- [torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): PyTorch official documentation
- [Convolution arithmetic guide](https://github.com/vdumoulin/conv_arithmetic): visual animations of convolution, stride, and padding
