## Denoising blindspot network

This is a reimplementation of the 2019 Nvidia paper: [High-Quality Self-Supervised Deep Image Denoising ](https://arxiv.org/abs/1901.10277). Since then many other papers have been published on the topic of self-supervised denoising, but this paper is still a good starting point for completely unsupervised, data-driven denoising.

The code has baseline implementations for RGB images and additive Gaussian noise. The high-power fields could also be added into the code
and the network could be trained on those multi-channel images. The loss function is able to handle that at `fp64`.

## Preliminary RGB results

![rgb_denoise_comparison](./examples/denoised_cifar10.png)