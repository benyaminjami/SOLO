# Enhanced SOLO: Instance Segmentation with Centerness Prediction

This repository contains modifications to the SOLO instance segmentation framework, introducing a centerness prediction branch and a mask quality constraint to refine instance detection accuracy and mask quality.

## Features
- **Centerness Prediction**: A dedicated branch for improved instance detection.
- **Mask Quality Constraint**: Filters out low-quality masks based on predefined criteria.

## Installation and Quick Start
For installation instructions and a quick start guide, please refer to the [mmdetection repository](https://github.com/open-mmlab/mmdetection).

## Note
The model with the centerness branch was fine-tuned, not trained from scratch. Training from the ground up may yield better results.
