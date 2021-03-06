# VTAMIQ: Transformers for Attention Modulated Image Quality Assessment

## Contents

This repository contains a Python implementation of our Full-Reference (FR) Image Quality Assessment (IQA)
method, Vision Transformer for Attention Modulated Image Quality (VTAMIQ), as described in:

<i> VTAMIQ: Transformers for Attention Modulated Image Quality Assessment.
Andrei Chubarau and James Clark. 2021,
<a href="https://arxiv.org/abs/2110.01655">Arxiv link</a>.
</i>

We provide training and testing code for various popular FR IQA datasets.

Please cite as:

```
@article{vtamiq2021,
  title={VTAMIQ: Transformers for Attention Modulated Image Quality Assessment},
  author={Chubarau, Andrei and Clark, James},
  journal = {CoRR},
  volume = {abs/2110.01655},
  year={2021},
  url = {https://arxiv.org/abs/2110.01655}
}
```

## Paper Abstract

Following the major successes of self-attention and Transformers for image analysis, 
we investigate the use of such attention mechanisms in the context of Image Quality Assessment (IQA) 
and propose a novel full-reference IQA method, Vision Transformer for Attention Modulated 
Image Quality (VTAMIQ). Our method achieves competitive or state-of-the-art performance on the 
existing IQA datasets and significantly outperforms previous metrics in cross-database evaluations. 
Most patch-wise IQA methods treat each patch independently; this partially discards global information 
and limits the ability to model long-distance interactions. We avoid this problem altogether by 
employing a transformer to encode a sequence of patches as a single global representation, 
which by design considers interdependencies between patches. We rely on various attention 
mechanisms -- first with self-attention within the Transformer, and second with channel 
attention within our difference modulation network -- specifically to reveal and enhance 
the more salient features throughout our architecture. With large-scale pre-training for 
further demonstrating the strength of transformer-based networks for vision modelling.

## VTAMIQ Summary

<img src='https://github.com/ch-andrei/VTAMIQ/blob/main/figures/vtamiq_diagram_compact.png' width=1200>

VTAMIQ uses a Vision Transformer (ViT) [1] modified for sparse patch extraction
to encode each input image as a single latent representation, computes the difference 
between the encoded representations for the reference and the distorted images, 
intelligently modulates this difference (DiffNet), and finally, 
interprets the modulated difference vector as an image quality score.

### Vision Transformer (ViT)

Note that we did not implement or train ViT but re-used a publicly
available pre-trained model from [https://github.com/jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch ).

VTAMIQ uses the "Base" (B_16) variant of ViT. Pretrained weights for ViT can be acquired via

```
# imagenet21k pre-train + imagenet2012 fine-tuning
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16.npz
```

and must be made available as ./modules/Attention/ViT/weights/imagenet21k+imagenet2012_ViT-B_16.npz

### Context-Aware Patch Sampling (CAPS)

As we use sparse sampling of patches, selecting more salient patches may contribute to more effective IQA. 
To this end, we leverage several properties of IQA and Human Visual System in our "context-aware" 
patch sampling (CAPS) scheme. CAPS allows us to train VTAMIQ faster and with fewer patches.

CAPS relies on probabilistic sampling to extract N patches from the input images.
We define pixelwise sampling probabilities based on a normalized weighted sum of three components:
i) a measure of perceptual difference between the reference and the test images (e.g., MSE, SSIM),
ii) centerbias, iii) uniform. We then rescale the components to range [0,1] and compute a weighted average with 
configurable weight parameters.

We further implement stratified sampling to decrease the variance and improve the uniformity of the extracted patches. 
The sampling probability is discretized into a grid of KxK blocks; block size is adaptive to input image resolution.
Each block is then allocated an appropriate number of samples given the combined probability.

We showcase CAPS with 50, 500, and 5000 patches below:
<img src='https://github.com/ch-andrei/VTAMIQ/blob/main/figures/patch_sampling-singlescale.png' width=1200>

To allow for IQA for high resolution images, we may further extract patches at multiple scales. 
The sampling scheme is applied for each scale independently. We first sample at the original resolution, 
then downscale and sample again. This procedure can be repeated until the desired number of scales is used.
<img src='https://github.com/ch-andrei/VTAMIQ/blob/main/figures/patch_sampling-multiscale.png' width=1200>

Note that we reduce the number of patches for larger scales such that the overall pixel coverage is approximately
uniform for every scale. For instance, using 16x16, 32x32, and 64x64: a single 32x32 patch contains 4 16x16 patches; 
64x64 contains 4 32x32 or 16 16x16 patches; therefore, we set the patch counts as per the ratio 16:4:1 to match the 
number of pixels for each scale.

See ./data/patch_sampling.py for more information.

### ViT modified for CAPS

We modify positional embedding for ViT to suit our patch sampling scheme. 
Unlike original ViT, we do not uniformly tile the input images instead opting to randomly
sample patches. As a result, the selected set is unordered and may contain overlapping patches.
To allow for full reuse of pre-trained ViT, we use the positional <i>uv</i> coordinate of each patch 
(known at the time of sampling), to index into the array of positional embeddings.

<img src='https://github.com/ch-andrei/VTAMIQ/blob/main/figures/vtamiq_vit_diagram_compact.png' width=400>

We complement positional embeddings with Scale Embeddings (SE) following the strategy presented by MUSIQ 
(Multi-scale Image Quality Transformer) [3]. We extract multi-scale patches as described above, and downsample 
to 16x16 resolution to allow all patches to be processed in parallel (by ViT) as a single sequence.

The final embeddings (before passing through the transformer) consist of the sum of Patch, Positional, and Scale embeddings. 

## Training Details

For our best performance, we train VTAMIQ in several stages.

Firstly, we use a Vision Transformer (ViT) pre-trained on ImageNet. 

We then train VTAMIQ on the large KADIS-700k [2] dataset, 
which contains 700,000 image pairs along with the corresponding scores given by 
11 conventional IQMs - this is our baseline model "weakly" pre-trained for IQA.

VTAMIQ is then fine-tuned on other IQA datasets and the performance is assessed.

## Training Code

Our training code is presented in ./run_main.py while training configuration file ./run_config.py 
contains various constants and parameters used throughout our code. 

The following files contain code for various training runs:
1. ./run_custom.py for a single run of training with custom parameters
2. ./run_multi.py for multiple runs of training (including result analysis, e.g. average performance etc.)

run_custom.py examplifies how we customize training parameters and train our model.

run_multi.py allows us to run multiple custom runs of training, record various train/validation/test metrics, 
and assess the final performance of our model.

## Pre-trained Model

[WIP]

April 9, 2022: <a href="https://drive.google.com/file/d/1ucEKoSOIyMn1fYpqOAG-8DtedVFg_i3-">Pretrained on KADIS-700k*</a> 

*We trained VTAMIQ on KADID10k, then used this trained model to predict quality scores for KADIS700k. These "improved"
KADIS700k scores were then used for training VTAMIQ from scratch. 
Note: the resulting model has 0.962 SROCC on KADID10k without being directly trained on it.

## References:

[1] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is
worth 16x16 words: Transformers for image recognition at scale, 2020.

[2] Hanhe Lin, Vlad Hosu, and Dietmar Saupe. DeepFL-IQA: Weak supervision for deep iqa feature learning. arXiv
preprint arXiv:2001.08113, 2020.

[3] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, Feng Yang. MUSIQ: Multi-scale Image Quality Transformer. 
arXiv:2108.05997, 2021.
