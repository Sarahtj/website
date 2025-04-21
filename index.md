---
title: Home
layout: home
nav_order: 1
---

# Refining ViTPose: A Transformer-Based Encoder-Decoder Framework with Feature Pyramids for Human Pose Estimation
{: .fs-9 }

ROB 499/599 Deep Learning for Robot Perception

Niva Ranavat, Sarah Jamil, Adithya Raman, Jacob Klinger

---

[Vitpose Documentation][vitpose]{: .btn .fs-5 .mb-4 .mb-md-0 }
[View it on GitHub][vitpose extension repo]{: .btn .fs-5 .mb-4 .mb-md-0 }

---
We conducted an ablation study on the original ViTPose architecture by fine-tuning it to incorporate a Feature Pyramid Network (FPN), aiming to enhance the accuracy of Human Pose Estimation. One limitation of the original ViTPose is its tendency to overlook fine-grained details and smaller features, particularly in scenes where objects are close together or overlapping. By integrating FPN, we aim to address this issue by improving multi-scale feature representation.

---

## Background

Human pose estimation is a fundamental task in computer vision with applications in activity recognition, animation, and human-computer interaction. Recent approaches like ViTPose leverage Vision Transformers to capture long-range dependencies and achieve impressive results. However, pure transformer-based models often lack multi-scale spatial detail critical for precise keypoint localization. To address this, we enhance ViTPose by incorporating a Feature Pyramid Network (FPN) and an encoder-decoder structure, enabling richer spatial representations and refined heatmap predictions for improved pose estimation accuracy. As a stretch goal, we explore extending the model to predict future human poses from observed sequences to enable motion forecasting.

--- 

## Dataset
We used the MS COCO dataset. The MS COCO dataset for human pose estimation provides images with rich annotations of keypoints for multiple people, including 17 body joints such as elbows, knees, and ankles. It was used in the original ViTPose and is also  widely used as a benchmark for evaluating human pose estimation models due to its diverse and challenging real-world scenes.

[View the dataset][coco dataset]{: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Network Architecture and Setup

Inputs:
- ThreeDPW Dataset Loader: Custom data loader tailored for the 3D Poses in the Wild (ThreeDPW) dataset
Modules:
- Data Preprocessing Module:
--Handles data normalization, augmentation, and preparation for training and evaluation
--Ensures compatibility with the model's input requirements
-Pose Estimation Model:
--Implements the core architecture for predicting human poses from input data
--Utilizes deep learning techniques to infer 3D joint positions
-Visualization Tools:
--Scripts and utilities for visualizing predicted poses against ground truth
--Aids in qualitative assessment of model performance
Outputs:
-Refined 3D Human Poses:
--3D joint positions representing human poses in various scenarios
--Suitable for downstream tasks such as action recognition or animation


---
## Results

---

[vitpose extension repo]: https://github.com/nranavat1/Refined_Human_Pose_Estimation
[vitpose]: https://arxiv.org/abs/2204.12484
[coco dataset]: https://cocodataset.org/#home
