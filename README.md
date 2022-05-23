#                feature detection and matching algorithm models



## Introduction		

This warehouse mainly uses C++ to compare traditional image feature detection and matching, and deep learning feature detection and matching algorithm models. Deep learning includes superpoint-superglue, and traditional algorithms include zkaze, surf, ORB, etc.

## Dependencies

OpenCV >= 3.4

CUDA >=10.2

CUDNN>=8.02

TensorRT>=7.2.3

## Contents

1. akaze feature point detection and matching display.

##### image pair

![akaze-image](./image/akaze-image.png)

##### camera

<video src="./image/akaze-video.mp4"></video>

2. superpoint-superpoint feature point detection and matching display.

##### image pair

![superglue-image](./image/superpoint-superglue-image.png)

##### camera

<video src="./image/superpoint-superglue-video.mp4"></video>

## reference

```
@inproceedings{sarlin20superglue,
  author    = {Paul-Edouard Sarlin and
               Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  title     = {{SuperGlue}: Learning Feature Matching with Graph Neural Networks},
  booktitle = {CVPR},
  year      = {2020},
  url       = {https://arxiv.org/abs/1911.11763}
}
```
