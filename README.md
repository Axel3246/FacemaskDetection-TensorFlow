# Facemask Detection using TensorFlow

This repository contains a TensorFlow-based implementation for detecting whether a person is wearing a face mask or not. The model is trained to classify images into two categories: **with mask** and **without mask**, achieving **87% accuracy** on the validation set.

## Overview
The goal is to build a CNN model to detect face masks in real-time, addressing challenges posed by the COVID-19 pandemic. The model uses TensorFlow and a custom dataset from Kaggle.

---

## Features
- Real-time detection (images, video, webcam).
- Pre-trained CNN with fine-tuning.
- Data preprocessing and augmentation support.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Axel3246/FacemaskDetection-TensorFlow.git
   cd FacemaskDetection-TensorFlow
---

## Dataset
- **Source**: [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) (7,553 images).
- **Classes**:
  - `with_mask`: 3,725 images.
  - `without_mask`: 3,828 images.

---

## Model Architecture
- **Layers**:
  - Input: `224x224x3` RGB images.
  - Convolutional layers with `32`, `64`, and `64` filters.
  - Max pooling and dropout for regularization.
  - Dense layer with `64` units (ReLU activation).
  - Output layer with sigmoid activation.
- **Optimizer**: Adam.
- **Loss Function**: Binary cross-entropy.

---

## Training
1. Split dataset into `80% train` and `20% test`.
2. Configure hyperparameters in `config.py`.
3. Run training:
   ```bash
   python train.py
   ```

---

## Results
- **Accuracy**: 87% on test set with a 0.34 loss.

---


## References
- Dataset: [Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- TensorFlow Documentation: [Data Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)
