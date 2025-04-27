---
title: Home
layout: home
nav_order: 1
---
# Refining ViTPose: A Transformer-Based Encoder-Decoder Framework with Feature Pyramids for Human Pose Estimation
{: .fs-9 }

ROB 499/599 Deep Learning for Robot Perception

Niva Ranavat, Sarah Jamil, Adithya Raman, Jacob Klinger

[Vitpose Documentation][vitpose]{: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View it on GitHub][vitpose extension repo]{: .btn .fs-5 .mb-4 .mb-md-0 }

---
We conducted an ablation study on the original ViTPose architecture by fine-tuning it to incorporate a Feature Pyramid Network (FPN), aiming to enhance the accuracy of Human Pose Estimation. One limitation of the original ViTPose is its tendency to overlook fine-grained details and smaller features, particularly in scenes where objects are close together or overlapping. By integrating FPN, we aim to address this issue by improving multi-scale feature representation.
In addition to this, we extended ViTPose by designing a secondary pose prediction network. In this framework, ViTPose is first used as a feature extractor on sequences of video frames, and a lightweight transformer model is then trained to predict the future poses across multiple frames. This extension allowed us to explore not just static pose estimation, but the task of forecasting human motion over time based on visual input.

---
## Background

Human pose estimation is a fundamental task in computer vision with applications in activity recognition, animation, and human-computer interaction. Recent approaches like ViTPose leverage Vision Transformers to capture long-range dependencies and achieve impressive results. However, pure transformer-based models often lack multi-scale spatial detail critical for precise keypoint localization. To address this, we enhance ViTPose by incorporating a Feature Pyramid Network (FPN) and an encoder-decoder structure, enabling richer spatial representations and refined heatmap predictions for improved pose estimation accuracy. As a stretch goal, we explore extending the model to predict future human poses from observed sequences to enable motion forecasting.

The original ViTPose model applies Vision Transformers (ViTs) to the task of human pose estimation by treating each image as a sequence of patches and processing them through self-attention layers to capture long-range spatial dependencies. Unlike traditional convolution-based methods, ViTPose leverages global context to accurately predict keypoints, achieving state-of-the-art results on benchmarks like COCO and MPII. It uses a simple yet effective backbone and a linear head to directly regress heatmaps for each keypoint, demonstrating that transformer-based models can excel in dense prediction tasks when properly adapted.

Predicting future human poses extends traditional pose estimation into the temporal domain, aiming to model and forecast human motion dynamics. This task requires learning not only spatial configurations of the body but also how these configurations evolve over time. Approaches for pose prediction typically involve sequence modeling techniques such as recurrent neural networks, temporal convolutions, or transformers. Accurate future pose forecasting has important applications in action anticipation, autonomous systems, and augmented reality, where understanding human motion trajectories is critical.

--- 
## Dataset
We used the MS COCO dataset. The MS COCO dataset for human pose estimation provides images with rich annotations of keypoints for multiple people, including 17 body joints such as elbows, knees, and ankles. It was used in the original ViTPose and is also  widely used as a benchmark for evaluating human pose estimation models due to its diverse and challenging real-world scenes.

For motion forecasting experiments, we utilized the 3D Poses in the Wild (3DPW) dataset. The 3DPW dataset contains real-world video sequences with accurate 3D and 2D pose annotations, captured from wearable IMUs and synchronized camera footage, making it valuable for studying human motion over time.

[View the coco dataset][coco dataset]{: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View the 3dpw dataset][3d poses in the wild]{: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }

---
## Network Architecture and Setup

### Original VitPose:

![](1.jpg)

Inputs:
- RGB Images:
  - Standard 2D images, typically from datasets like COCO or MPII
  - Resized and normalized to match model requirements
-  2D Keypoint Annotations (during training):
  - Supervised keypoint labels for calculating heatmap-based loss

Modules:
- Patch Embedding Module:
  - Divides the input image into fixed-size patches and flattens them
  - Projects patch vectors into a consistent embedding space
- Vision Transformer Encoder:
  - Applies multi-head self-attention to model global dependencies across all patches
  - Includes positional embeddings and a learnable class token
- Pose Prediction Head:
  - A lightweight convolutional or linear layer that converts encoded features into heatmaps
  - One heatmap per keypoint

Outputs:
- 2D Keypoint Heatmaps:
  - Predicted probability maps indicating the location of each joint
  - Output shape typically (N, num_keypoints, H, W)
- Keypoint Coordinates (post-processing):
  - Extracted joint locations from heatmaps via argmax or soft-argmax
  - Used for evaluation metrics such as PCK (Percentage of Correct Keypoints)

![](2.jpg)

### Proposed Change:
Inputs:
- ThreeDPW Dataset Loader:
  - Custom data loader tailored for the 3D Poses in the Wild (ThreeDPW) dataset

Modules:
- Data Preprocessing Module:
  - Handles data normalization, augmentation, and preparation for training and evaluation
  - Ensures compatibility with the model's input requirements
- Pose Estimation Model:
  - Implements the core architecture for predicting human poses from input data
  - Utilizes deep learning techniques to infer 3D joint positions
- Visualization Tools:
  - Scripts and utilities for visualizing predicted poses against ground truth
  - Aids in qualitative assessment of model performance

Outputs:
- Refined 3D Human Poses:
  - 3D joint positions representing human poses in various scenarios
  - Suitable for downstream tasks such as action recognition or animation

### Future Pose Predictions:

![](image-6.png)
Example of frame images from the 3D Poses in the Wild dataset

Inputs:
- 2D Keypoint Sequences:
  - Normalized 2D keypoints extracted from RGB images using ViTPose
  - Represented as sequences of 10 frames (input window)

Modules:
- Input Projection Layer:
  - Applies a linear transformation to project each 2D keypoint vector into a hidden feature space
  - Enables compatibility with transformer input expectations
- Transformer Encoder:
  - Models temporal dependencies across the sequence of frames
  - Uses multi-head self-attention to capture how poses evolve over time
- Pose Prediction Decoder:
  - A linear layer that maps encoded features back into predicted 2D keypoint coordinates
  - Predicts multiple future frames (typically 5)

Outputs:
- Predicted 2D Keypoint Sequences:
  - 5 frames of predicted future keypoints
  - Each frame output is a set of (x, y) coordinates for each joint

---
## Results

### Feature Pyramid Network

### Future Pose Prediction

Examples of predicted keypoints from the extended Vitpose Predictor:

![](image.png)

![](image-5.png)

To extend pose estimation into the temporal domain, we designed a lightweight pose prediction network that predicts future human poses from observed keypoint sequences. Starting from 2D keypoints extracted using ViTPose, a linear projection first maps the input into a hidden feature space. A Transformer encoder then models the temporal relationships across the 10 input frames, capturing how human motion evolves over time. Finally, a simple decoder projects these temporal features into predicted 2D keypoints for 5 future frames. The model is trained by minimizing the L2 distance between predicted and ground-truth keypoints, enabling it to learn motion patterns and anticipate future poses.

---

[vitpose extension repo]: https://github.com/nranavat1/Refined_Human_Pose_Estimation
[vitpose]: https://arxiv.org/abs/2204.12484
[coco dataset]: https://cocodataset.org/#home
[3d poses in the wild]: https://www.google.com/search?client=safari&rls=en&q=3d+poses+in+the+wild&ie=UTF-8&oe=UTF-8
