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
<details class="toc-details" markdown="1">
<summary><b>Table of Contents</b></summary>

* TOC
{:toc}

</details>


`torch.nn.Conv2d` applies a 2D convolution over an input signal (e.g., an image). A kernel (sliding window) moves across the input, computing a weighted sum at each position to produce a smaller or same-sized output.

**Why and when**: Conv2d is the core building block of CNNs. It **downsamples** spatial dimensions (when stride > 1) while learning spatial features. Used in encoders, feature extractors, and classification networks.

```
Conv2d (stride=2): shrinks spatial dimensions

Input (6x6)              Output (3x3)
+--+--+--+--+--+--+     +--+--+--+
|  |  |  |  |  |  |     |  |  |  |
+--+--+--+--+--+--+     +--+--+--+
|  |  |  |  |  |  | --> |  |  |  |
+--+--+--+--+--+--+     +--+--+--+
|  |  |  |  |  |  |     |  |  |  |
+--+--+--+--+--+--+     +--+--+--+
|  |  |  |  |  |  |
+--+--+--+--+--+--+
|  |  |  |  |  |  |
+--+--+--+--+--+--+
|  |  |  |  |  |  |
+--+--+--+--+--+--+
```

## Conv2d (2D Convolution / Downsampling)

### `Conv2d` API Signature

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

### Example: `Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)`

- **1** input channel (grayscale image)
- **32** output channels (32 learned 3x3 filters)
- **3x3** kernel: each filter is a 3x3 sliding window that computes a weighted sum over a 3x3 patch
- **stride=2**: the kernel jumps 2 pixels per step, halving spatial dimensions, i.e. downsampling
- **padding=1**: 1 pixel of zeros added on each side, preventing excessive shrinkage

### Output Size Formula

```
H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1
W_out = floor((W_in + 2*padding - kernel_size) / stride) + 1
```

#### Derivation

The formula counts how many times the kernel fits as it slides across the padded input:

1. **Pad the input**: effective size = `H_in + 2*padding`
2. **First kernel placement uses up `kernel_size` pixels**: remaining pixels to slide over = `H_in + 2*padding - kernel_size`
3. **Each step moves `stride` pixels**: number of additional placements = `floor(remaining / stride)`
4. **+1 for the first placement**: total = `floor((H_in + 2*padding - kernel_size) / stride) + 1`

#### Worked example

Input `1x28x28` through `Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)`:

```
H_out = floor((28 + 2*1 - 3) / 2) + 1
      = floor(27 / 2) + 1
      = 13 + 1
      = 14
```

Output shape: `32x14x14` (32 channels, each 14x14).

#### Common cases

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

### What Each Filter Learns

Each filter is a 3D block of learnable weights shaped `in_channels x kernel_size x kernel_size`. It must match the input's channel depth so it can compute a dot product across **all channels simultaneously** at each spatial position.

For `Conv2d(in_channels=3, out_channels=32, kernel_size=3)` (RGB input, 32 filters, 3x3 kernel):

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
#### Numerical example (slide + dot product)

Simplified single-channel example (in_channels=1, 3x3 input, 2x2 kernel):

```
Input (3x3):           Kernel (2x2):
[1, 2, 3]             [a, b]
[4, 5, 6]             [c, d]
[7, 8, 9]

Step 1: kernel at (0,0)              Step 2: kernel at (0,1)
dot product over [1,2,4,5]:          dot product over [2,3,5,6]:
  1*a + 2*b + 4*c + 5*d               2*a + 3*b + 5*c + 6*d

Step 3: kernel at (1,0)              Step 4: kernel at (1,1)
dot product over [4,5,7,8]:          dot product over [5,6,8,9]:
  4*a + 5*b + 7*c + 8*d               5*a + 6*b + 8*c + 9*d

Output (2x2):
[1*a+2*b+4*c+5*d,   2*a+3*b+5*c+6*d]
[4*a+5*b+7*c+8*d,   5*a+6*b+8*c+9*d]
```

Key insight: 3x3 input → 2x2 output (spatial size **shrank**).


- A single filter spans the **full channel depth** of the input. It doesn't convolve over one channel at a time. This is how Conv2d learns cross-channel patterns (e.g., a filter that detects edges where red meets green).
- Early layers learn low-level features (edges, textures); deeper layers learn higher-level patterns.
- The total number of learnable parameters: `out_channels * (in_channels * kernel_size^2 + 1)` (the +1 is the bias per filter, if `bias=True`).

## ConvTranspose2d (Transposed Convolution / Upsampling)

`torch.nn.ConvTranspose2d` is the spatial reverse of `Conv2d`: it **increases** spatial dimensions (upsampling).

**Why and when**: used in decoders to upsample back to full resolution. Autoencoders, VAEs, U-Nets, and video tokenizers (like GAIA-2) encode into a small latent space (downsample with Conv2d), then decode back (upsample with ConvTranspose2d).

```
ConvTranspose2d (stride=2): grows spatial dimensions

Input (3x3)              Zero-inserted (5x5)         Output (6x6)
+--+--+--+               +--+--+--+--+--+            +--+--+--+--+--+--+
|1 |2 |3 |               |1 | 0|2 | 0|3 |            |  |  |  |  |  |  |
+--+--+--+   insert      +--+--+--+--+--+   Conv2d   +--+--+--+--+--+--+
|4 |5 |6 | → zeros   →   | 0| 0| 0| 0| 0|  -----→    |  |  |  |  |  |  |
+--+--+--+   between     +--+--+--+--+--+            +--+--+--+--+--+--+
|7 |8 |9 |   elements    |4 | 0|5 | 0|6 |            |  |  |  |  |  |  |
+--+--+--+               +--+--+--+--+--+            +--+--+--+--+--+--+
                         | 0| 0| 0| 0| 0|            |  |  |  |  |  |  |
                         +--+--+--+--+--+            +--+--+--+--+--+--+
                         |7 | 0|8 | 0|9 |            |  |  |  |  |  |  |
                         +--+--+--+--+--+            +--+--+--+--+--+--+
                                                     |  |  |  |  |  |  |
                                                     +--+--+--+--+--+--+
```

### `ConvTranspose2d` API Signature

```python
nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True)
```

| Parameter | Meaning |
|---|---|
| `in_channels` | Number of channels in the input (from encoder output) |
| `out_channels` | Number of filters (each produces one output channel) |
| `kernel_size` | Size of the kernel stamped at each position (e.g., `3` means 3x3) |
| `stride` | Zeros inserted between input elements: `stride - 1` zeros (default 1 = no insertion) |
| `padding` | Pixels trimmed from each side of the output (default 0) |
| `output_padding` | Extra rows/columns added to resolve stride ambiguity (default 0) |
| `bias` | Whether to add a learnable bias to each output channel (default True) |

### Example: `ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)`

- **64** input channels (from encoder's output)
- **32** output channels (fewer channels as we upsample spatially)
- **3x3** kernel: each filter is a 3x3 window stamped at each input position
- **stride=2**: inserts 1 zero between each input element, doubling spatial size
- **padding=1**: trims 1 pixel from each side of the output
- **output_padding=1**: adds 1 extra row/column to resolve the stride ambiguity

### Output Size Formula

```
H_out = (H_in - 1) * stride - 2*padding + kernel_size + output_padding
W_out = (W_in - 1) * stride - 2*padding + kernel_size + output_padding
```

#### Derivation

ConvTranspose2d is the spatial reverse of Conv2d, so the formula inverts the Conv2d output size formula:

1. **Start from Conv2d's formula**: `H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1`
2. **Solve for `H_in`** (the original input that would produce `H_out` via Conv2d): rearrange to get the input size back
3. **Remove the floor**: `(H_out - 1) * stride` undoes the division-and-floor, recovering the number of pixels the kernel traversed
4. **Add back the kernel footprint**: `+ kernel_size` accounts for the first kernel placement
5. **Subtract padding**: `- 2 * padding` removes the padding that was added
6. **Resolve ambiguity**: `+ output_padding` selects which of the valid sizes to produce (since floor lost information)

Result: `H_out = (H_in - 1) * stride - 2*padding + kernel_size + output_padding`

#### Worked example

Input `64x7x7` through `ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)`:

```
H_out = (7 - 1) * 2 - 2*1 + 3 + 1
      = 12 - 2 + 3 + 1
      = 14
```

Output shape: `32x14x14` (32 channels, each 14x14). Used in the VAE decoder in [VAE with ELBO](/re-log/vae-with-elbo/).

#### Common cases

| Configuration | Effect on spatial size |
|---|---|
| `kernel=3, stride=2, padding=1, output_padding=1` | Doubles size (exact inverse of stride-2 Conv2d) |
| `kernel=4, stride=2, padding=1` | Doubles size (alternative, no output_padding needed) |
| `kernel=2, stride=2, padding=0` | Doubles size (minimal kernel) |

### What is `output_padding`?

Conv2d with `stride=2` uses `floor()`, so different input sizes can produce the same output:

```
Conv2d(kernel_size=3, stride=2, padding=1):
  14x14 input → floor((14 + 2 - 3) / 2) + 1 = floor(13/2) + 1 = 7   ← floor drops 0.5
  13x13 input → floor((13 + 2 - 3) / 2) + 1 = floor(12/2) + 1 = 7   ← exact division
```

Both 13x13 and 14x14 produce 7x7. So when ConvTranspose2d tries to go back from 7x7, which should it produce?

```
ConvTranspose2d(kernel_size=3, stride=2, padding=1):
  output_padding=0 → (7-1)*2 - 2 + 3 + 0 = 13   ← recovers 13x13
  output_padding=1 → (7-1)*2 - 2 + 3 + 1 = 14   ← recovers 14x14
```

`output_padding` resolves this ambiguity. It does **not** add zero-padding: it just selects which of the valid output sizes to produce. Its value must be less than `stride`.

### What Each Filter Does

Each filter is a 3D block of learnable weights shaped `in_channels x kernel_size x kernel_size`. Instead of gathering a patch into a scalar (like Conv2d), it **scatters** each scalar into a kernel-sized patch.

For `ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3)` (64-channel input, 32 output channels, 3x3 kernel):

```
One filter shape: 64 x 3 x 3  (in_channels x kernel_h x kernel_w)
                  = 576 weights

At each input spatial position:
  - takes 1 scalar from each of the 64 input channels
  - multiplies each scalar by its own 3x3 kernel
  - sums the 64 resulting 3x3 patches into one 3x3 patch
  - stamps this 3x3 patch onto the output grid
  - where stamps overlap, values are summed

One filter stamping across all positions → 1 output channel (a 2D feature map)
32 filters → 32 output channels

Compare with Conv2d:
  Conv2d:          slides kernel, GATHERS each 3x3 patch into 1 scalar (shrinks)
  ConvTranspose2d: at each position, SCATTERS 1 scalar into a 3x3 patch (grows)
```

#### Numerical example (stamp + overlap + sum)

Simplified single-channel example (in_channels=1, stride=1, padding=0, output_padding=0, 2x2 input, 3x3 kernel):

```
Output grid size from formula: H_out = (2-1)*1 - 2*0 + 3 + 0 = 4  → 4x4

Input (2x2):       Kernel (3x3):
[1, 2]             [a, b, c]
[3, 4]             [d, e, f]
                   [g, h, i]

Each step stamps a 3x3 kernel onto the 4x4 output grid.
Zeros fill positions where this stamp has no contribution.

Step 1: input=1 at (0,0)        Step 2: input=2 at (0,1)
stamp 1*kernel at (0,0):        stamp 2*kernel at (0,1):

[1*a, 1*b, 1*c,  0]             [0, 2*a, 2*b, 2*c]
[1*d, 1*e, 1*f,  0]             [0, 2*d, 2*e, 2*f]
[1*g, 1*h, 1*i,  0]             [0, 2*g, 2*h, 2*i]
[0,  0,  0,  0]                 [0,  0,  0,  0]

Step 3: input=3 at (1,0)        Step 4: input=4 at (1,1)
stamp 3*kernel at (1,0):        stamp 4*kernel at (1,1):

[0,  0,  0,  0]                 [0,  0,  0,  0]
[3*a, 3*b, 3*c,  0]             [0, 4*a, 4*b, 4*c]
[3*d, 3*e, 3*f,  0]             [0, 4*d, 4*e, 4*f]
[3*g, 3*h, 3*i,  0]             [0, 4*g, 4*h, 4*i]

Output (4x4) = sum of all 4 stamps:

[1*a,      1*b+2*a,         1*c+2*b,         2*c     ]
[1*d+3*a,  1*e+2*d+3*b+4*a, 1*f+2*e+3*c+4*b, 2*f+4*c ]
[1*g+3*d,  1*h+2*g+3*e+4*d, 1*i+2*h+3*f+4*e, 2*i+4*f ]
[3*g,      3*h+4*g,         3*i+4*h,         4*i     ]
```

Key insight: 2x2 input → 4x4 output (spatial size **grew**).

- The total number of learnable parameters: `in_channels * (out_channels * kernel_size^2) + out_channels` (bias per output channel, if `bias=True`). 
- Note: the weight shape is `(in_channels, out_channels, kernel_h, kernel_w)`, **transposed** relative to Conv2d's `(out_channels, in_channels, kernel_h, kernel_w)`.

### How It Actually Works (Not "Deconvolution")

Despite the name "transposed convolution," it is **not** the inverse of convolution. It:
1. Inserts `stride - 1` zeros between each input element (upsampling by insertion)
2. Applies a standard convolution on the zero-inserted input
3. The result has larger spatial dimensions than the input

This is equivalent to multiplying by the **transpose** of the Conv2d weight matrix (hence the name), but the learned weights are independent.

### 3D Variant: ConvTranspose3d

For video (temporal + spatial), GAIA-2 uses 3D transposed convolutions with stride `(T, H, W)`:
- `stride=(2, 8, 8)`: upsample 2x in time, 8x in each spatial dimension
- Same output size formula applies independently per dimension:

```
T_out = (T_in - 1) * stride_t - 2*padding_t + kernel_size_t + output_padding_t
H_out = (H_in - 1) * stride_h - 2*padding_h + kernel_size_h + output_padding_h
W_out = (W_in - 1) * stride_w - 2*padding_w + kernel_size_w + output_padding_w
```

---

## References

- [torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): PyTorch official documentation
- [torch.nn.ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html): PyTorch official documentation
- [Convolution arithmetic guide](https://github.com/vdumoulin/conv_arithmetic): visual animations of convolution, stride, padding, and transposed convolution
