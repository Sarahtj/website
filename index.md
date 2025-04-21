---
title: Home
layout: home
nav_order: 1
---

# Refining ViTPose: A Transformer-Based Encoder-Decoder Framework with Feature Pyramids for Human Pose Estimation
{: .fs-9 }

ROB 499/599 Deep Learning for Robot Perception

Niva Ranavat, Sarah Jamil, Adithya Raman, Jacob Klinger

[Vitpose Documentation][vitpose]{: .btn .fs-5 .mb-4 .mb-md-0 }
[View it on GitHub][vitpose extension repo]{: .btn .fs-5 .mb-4 .mb-md-0 }

---
We conducted an ablation study on the original ViTPose architecture by fine-tuning it to incorporate a Feature Pyramid Network (FPN), aiming to enhance the accuracy of Human Pose Estimation. One limitation of the original ViTPose is its tendency to overlook fine-grained details and smaller features, particularly in scenes where objects are close together or overlapping. By integrating FPN, we aim to address this issue by improving multi-scale feature representation.

---

## Background

Human pose estimation is a fundamental task in computer vision with applications in activity recognition, animation, and human-computer interaction. Recent approaches like ViTPose leverage Vision Transformers to capture long-range dependencies and achieve impressive results. However, pure transformer-based models often lack multi-scale spatial detail critical for precise keypoint localization. To address this, we enhance ViTPose by incorporating a Feature Pyramid Network (FPN) and an encoder-decoder structure, enabling richer spatial representations and refined heatmap predictions for improved pose estimation accuracy. As a stretch goal, we explore extending the model to predict future human poses from observed sequences to enable motion forecasting.

--- 

## Dataset

---

## Network Architecture and Setup

---
## Results

---

[vitpose extension repo]: https://github.com/nranavat1/Refined_Human_Pose_Estimation
[vitpose]: https://arxiv.org/abs/2204.12484
