# Technical Details

This document provides an in-depth explanation of the machine learning pipeline, model architecture, training methodology, optimization techniques, and Android deployment workflow used in the **SkinDiseaseCheck** application.

---

# Table of Contents

- Project Objective
- Dataset
- Data Preparation
- Image Preprocessing
- Data Augmentation
- Model Architecture
- Training Configuration
- Regularization Techniques
- Hyperparameter Optimization
- TensorFlow Lite Conversion
- Android Inference Pipeline
- Supported Disease Classes
- Model Performance
- Technical Design Decisions
- Current Limitations
- Future Enhancements

---

# Project Objective

The primary objective of this project is to demonstrate the deployment of a Deep Learning image classification model on Android devices using TensorFlow Lite.

The application enables users to classify common skin diseases directly on their mobile devices without requiring an internet connection.

Key objectives include:

- Offline inference
- Fast prediction
- Privacy-preserving computation
- Lightweight mobile deployment

---

# Dataset

**Source**

Kaggle

Dataset:

```
Skin Diseases Image Dataset
```

Dataset Provider:

```
ismailpromus/skin-diseases-image-dataset
```

The dataset contains images belonging to multiple common skin disease categories.

---

# Data Preparation

The dataset was organized into the following directory structure:

```
Dataset/

├── Train/
└── Test/
```

The training dataset was further divided using:

```python
validation_split = 0.2
```

Resulting in:

- 80% Training Data
- 20% Validation Data

The independent test set was used for final evaluation.

---

# Image Preprocessing

Before training, every image underwent the following preprocessing steps:

- Resize to **224 × 224**
- RGB conversion
- Pixel normalization

Normalization was performed using:

```python
rescale = 1./255
```

This scales pixel values from:

```
0–255

↓

0–1
```

which improves numerical stability during training.

---

# Data Augmentation

Medical datasets are often limited and susceptible to overfitting.

To improve generalization, multiple augmentation techniques were applied during training.

The augmentation pipeline includes:

- Random Rotation
- Horizontal Flip
- Width Shift
- Height Shift
- Zoom
- Shear Transformation

These transformations generate diverse training samples without requiring additional data collection.

---

# Model Architecture

The deployed model is a custom Convolutional Neural Network (CNN).

Architecture:

```
Input Layer

↓

Conv2D

↓

MaxPooling

↓

Batch Normalization

↓

Conv2D

↓

MaxPooling

↓

Batch Normalization

↓

Conv2D

↓

MaxPooling

↓

Batch Normalization

↓

Flatten

↓

Dense Layer

↓

Dropout

↓

Softmax Output
```

The output layer predicts the probability distribution across all supported disease classes.

---

# Why Convolutional Neural Networks?

CNNs are particularly effective for image classification because they automatically learn hierarchical visual features.

Examples include:

- Edges
- Texture
- Color distribution
- Lesion boundaries
- Spatial patterns

Unlike traditional computer vision methods, CNNs eliminate the need for handcrafted feature engineering.

---

# Training Configuration

## Loss Function

Categorical Cross Entropy

Reason:

Suitable for multi-class classification problems where only one class represents the correct prediction.

---

## Optimizer

Adam Optimizer

Adam combines:

- Momentum
- Adaptive Learning Rates

Benefits include:

- Faster convergence
- Stable optimization
- Better performance than vanilla SGD for many image classification tasks

---

# Regularization Techniques

## Batch Normalization

Batch Normalization was added after convolutional layers.

Benefits:

- Faster convergence
- Improved gradient flow
- Reduced internal covariate shift
- Increased training stability

---

## Dropout

Dropout was applied before the final classification layer.

Purpose:

- Reduce overfitting
- Improve model generalization
- Encourage robust feature learning

---

# Hyperparameter Optimization

The project explored Hyperband using Keras Tuner.

Parameters investigated include:

- Number of convolution filters
- Dense layer size
- Learning rate

Although Hyperband was evaluated, the final deployed model uses the best-performing manually validated configuration to balance:

- Accuracy
- Model Size
- Mobile Inference Speed

---

# TensorFlow Lite Conversion

After model training, the Keras model was converted into TensorFlow Lite format.

Conversion pipeline:

```
.keras Model

↓

TensorFlow Lite Converter

↓

.tflite Model
```

Benefits of TensorFlow Lite:

- Smaller model size
- Faster inference
- Lower memory consumption
- Optimized mobile deployment

---

# Android Inference Pipeline

The Android application performs inference through the following workflow:

```
Gallery Image

↓

Bitmap Loading

↓

Resize (224×224)

↓

Normalize Pixel Values

↓

TensorBuffer Conversion

↓

TensorFlow Lite Interpreter

↓

Softmax Probability Output

↓

Highest Probability Class

↓

Prediction Display
```

All inference occurs locally on the device.

No network requests are required.

---

# Supported Disease Classes

The model currently predicts the following categories:

- Eczema
- Melanoma
- Atopic Dermatitis
- Basal Cell Carcinoma (BCC)
- Melanocytic Nevi
- Benign Keratosis-like Lesions (BKL)
- Psoriasis / Lichen Planus
- Seborrheic Keratoses
- Tinea / Ringworm / Fungal Infection
- Warts / Molluscum / Viral Infection

---

# Model Performance

| Metric | Value |
|---------|------:|
| Test Accuracy | ~70% |
| Input Resolution | 224 × 224 |
| Number of Classes | 10 |
| Deployment | TensorFlow Lite |

Performance may vary depending on:

- Dataset split
- Random initialization
- Image quality
- Lighting conditions

---

# Technical Design Decisions

## Why TensorFlow Lite?

TensorFlow Lite is specifically designed for edge devices.

Advantages:

- Optimized for Android
- Low latency
- Small binary size
- Low memory usage
- Offline execution

---

## Why On-device Inference?

Executing inference directly on the device offers several benefits.

- Improved privacy
- No cloud infrastructure required
- Faster response time
- Internet independence
- Reduced operational cost

---

## Why Image Augmentation?

Medical image datasets often contain limited samples.

Augmentation improves robustness against:

- Camera angle
- Rotation
- Illumination changes
- Zoom
- Minor positional variations

---

## Why Batch Normalization?

Batch Normalization improves training efficiency by stabilizing feature distributions.

This enables:

- Faster convergence
- Higher learning rates
- Improved optimization

---

## Why Dropout?

Dropout randomly disables neurons during training.

Benefits include:

- Reduced overfitting
- Improved generalization
- Better robustness on unseen images

---

# Current Limitations

Current limitations of the project include:

- Limited to ten disease classes
- Sensitive to poor lighting
- Dependent on image quality
- No lesion segmentation
- No multi-label prediction
- No severity estimation
- Educational use only

---

# Future Enhancements

Potential improvements include:

- EfficientNet-B0/B3
- MobileNetV3
- Vision Transformers (ViT)
- Transfer Learning
- Grad-CAM Explainability
- Confidence Thresholding
- Lesion Segmentation
- Camera-based Real-time Detection
- Disease Information Module
- Treatment Recommendation System
- Cloud Synchronization
- Medical Report Generation

---

# Disclaimer

This project has been developed solely for educational, research, and demonstration purposes.

The predictions generated by the application are based on patterns learned from the training dataset and should **not** be interpreted as medical advice or professional diagnosis.

Users should always consult qualified healthcare professionals for medical evaluation and treatment.