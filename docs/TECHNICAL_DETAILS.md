# Technical Details

This document provides an in-depth explanation of the machine learning pipeline, model architecture, training methodology, optimization techniques, input validation pipeline, and Android deployment workflow used in the **SkinDiseaseCheck** application.

---

# 📚 Table of Contents

1. [Project Objective](#project-objective)
2. [Dataset](#dataset)
3. [Data Preparation](#data-preparation)
4. [Image Preprocessing](#image-preprocessing)
5. [Data Augmentation](#data-augmentation)
6. [Model Architecture](#model-architecture)
7. [Training Configuration](#training-configuration)
8. [Regularization Techniques](#regularization-techniques)
9. [Hyperparameter Optimization](#hyperparameter-optimization)
10. [TensorFlow Lite Conversion](#tensorflow-lite-conversion)
11. [Input Validation Pipeline (New in v0.2.0)](#input-validation-pipeline-new-in-v020)
12. [Android Inference Pipeline](#android-inference-pipeline)
13. [Supported Disease Classes](#supported-disease-classes)
14. [Model Performance](#model-performance)
15. [Technical Design Decisions](#technical-design-decisions)
16. [Known Issues](#known-issues)
17. [Current Limitations](#current-limitations)
18. [Future Enhancements](#future-enhancements)
19. [Disclaimer](#disclaimer)

---

# Project Objective

The primary objective of this project is to demonstrate the deployment of a Deep Learning image classification model on Android devices using TensorFlow Lite.

The application enables users to classify common skin diseases directly on their mobile devices without requiring an internet connection.

Key objectives include:

- Offline inference
- Fast prediction
- Privacy-preserving computation
- Lightweight mobile deployment
- (v0.2.0) Reducing misleading predictions on invalid/non-skin input images

---

# Dataset

**Source**

Kaggle

Dataset:

Skin Diseases Image Dataset

Dataset Provider:
`
ismailpromus/skin-diseases-image-dataset
`

The dataset contains images belonging to multiple common skin disease categories.

> **Note:** The dataset consists exclusively of skin disease images and does not include a "non-skin" or "negative" class. This is the underlying reason the model itself cannot distinguish valid from invalid input — see [Input Validation Pipeline](#input-validation-pipeline-new-in-v020) for how this is addressed at the application layer instead.

---

# Data Preparation

The dataset was organized into the following directory structure:

`
Dataset/
    ├── Train/
    └── Test/
`

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

`0–255
↓
0–1`


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

```mermaid
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

_**Note:** This architecture is unchanged in v0.2.0. No retraining or model changes were made in this release — see [Input Validation Pipeline](#input-validation-pipeline-new-in-v020) for what was added instead._

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
```mermaid
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

# Input Validation Pipeline (New in v0.2.0)

## Background

The classification model was trained exclusively on skin disease images and has no concept of "not skin." In earlier versions (v0.1.0), any selected image — a face, a screenshot, an unrelated photo — was passed directly to the classifier, which would still output a confident-sounding prediction from its 10 fixed classes, regardless of relevance. This was misleading, since the model has no mechanism to say "none of these classes apply."

v0.2.0 addresses this at the **application layer**, using a sequence of lightweight, fully offline validation checks that run before an image reaches the classifier. Each stage can reject the image outright, showing the user a specific reason instead of a disease name.

## Pipeline Stages

### 1. Face Detection

- **Library:** ML Kit Face Detection (on-device, bundled model)
- **Purpose:** Rejects portraits, selfies, and any photo where a face is detected
- **Rationale:** Faces contain large areas of skin-toned pixels and can pass simple color-based checks, but are not relevant to skin disease classification
- **Result on match:** *"This looks like a face or portrait"*

### 2. Text Recognition (OCR)

- **Library:** ML Kit Text Recognition, Latin script (on-device, bundled model)
- **Purpose:** Rejects screenshots, documents, UI mockups, and other text-heavy images
- **Rationale:** Real photographs of skin essentially never contain readable rendered text, while screenshots and documents almost always do — this is a more reliable discriminator than color or texture alone for this specific failure case
- **Thresholds:** an image is rejected if it contains at least a minimum number of distinct text blocks, or at least a minimum total character count (either condition is sufficient)
- **Result on match:** *"This looks like a screenshot or document"*

### 3. Skin-Tone Heuristic

- **Method:** Color space analysis in YCbCr, sampled across a grid of pixels
- **Purpose:** Confirms the image contains a sufficient proportion of skin-like pixels before proceeding to classification
- **v0.2.0 change:** the acceptable Cb/Cr range was **widened** relative to v0.1.0 to also include inflamed, reddened, and scaly/pale skin tones, not just healthy baseline skin tone. Earlier, disease-affected skin (which often looks visually different from healthy skin) was sometimes being incorrectly rejected by this stage — the wider range corrects that false-rejection issue
- **Result on match (below threshold):** *"No skin detected"*

### 4. Confidence Thresholding (post-classification)

- Applied after the model produces its output, not before
- If the top predicted class's confidence falls below a minimum threshold, the result is shown as **"Uncertain result"** along with the raw confidence percentage, rather than presenting a specific disease name as if it were certain

## How This Changes Results Compared to v0.1.0

| Scenario                                        | v0.1.0 behavior                           | v0.2.0 behavior                                                         |
|-------------------------------------------------|-------------------------------------------|-------------------------------------------------------------------------|
| Photo of a face                                 | Confident disease prediction (misleading) | Rejected: "This looks like a face or portrait"                          |
| Screenshot / document                           | Confident disease prediction (misleading) | Rejected: "This looks like a screenshot or document"                    |
| Random non-skin object/scene                    | Confident disease prediction (misleading) | Rejected: "No skin detected"                                            |
| Genuine skin disease photo                      | Prediction shown regardless of confidence | Prediction shown if confident; "Uncertain result" + confidence % if not |
| Inflamed/diseased skin (v0.1.0 heuristic range) | Occasionally rejected as "no skin"        | Correctly passes the skin-tone check due to widened range               |

## Important Caveats

- All four stages are **heuristic or general-purpose ML Kit models**, not classifiers specifically trained on this app's skin disease domain. They improve reliability but are not guaranteed to be perfectly accurate in every case.
- A borderline image (e.g., very close-up skin with an incidental text watermark, or an image with ambiguous coloring) may occasionally be rejected even though a human would consider it valid input.
- Conversely, some non-skin images with skin-like coloring and no face/text may still pass through to the classifier undetected.
- These stages do not modify or retrain the classification model itself — they only control *which* images are allowed to reach it.

---

# Android Inference Pipeline

The Android application performs inference through the following workflow:

Gallery Image

↓

Face Detection (on-device, ML Kit)
reject if face found
↓

Text Recognition / OCR (on-device, ML Kit)
reject if text-heavy (screenshot/document)
↓

Skin-Tone Heuristic (YCbCr analysis)
reject if insufficient skin-like pixels
↓

Bitmap Resize (224×224)

↓

Normalize Pixel Values

↓

TensorBuffer Conversion

↓

TensorFlow Lite Interpreter

↓

Softmax Probability Output

↓

Confidence Thresholding
flag as "Uncertain result" if below threshold
↓

Prediction Display (label + confidence %)


All stages — including face detection and OCR — run locally on the device using bundled offline models.

No network requests are required at any point in this pipeline.

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

| Metric            |           Value |
|-------------------|----------------:|
| Test Accuracy     |            ~70% |
| Input Resolution  |       224 × 224 |
| Number of Classes |              10 |
| Deployment        | TensorFlow Lite |

Performance may vary depending on:

- Dataset split
- Random initialization
- Image quality
- Lighting conditions

**Note:** This accuracy figure describes the classifier's performance on valid skin images only, and is unaffected by the v0.2.0 validation pipeline changes — the pipeline changes what reaches the model, not the model's own learned accuracy.

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

## Why Add an Input Validation Pipeline Instead of Retraining the Model?

Retraining the CNN with an explicit "not skin" class would be the more robust long-term solution, but requires collecting and labeling a large, diverse negative dataset (non-skin photos across many categories), followed by full retraining and re-evaluation.

For v0.2.0, adding pre-trained, general-purpose ML Kit models (face detection, OCR) as a gating layer was a faster, lower-risk way to address the most common and visible failure modes (faces, screenshots) without touching the existing trained classifier. A dedicated "not skin" model class remains a planned future enhancement.

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

# Known Issues

- **Dark mode text rendering (fixed in v0.2.0 for most devices):** in v0.1.0, prediction text was rendered dark-on-dark on devices with system dark mode enabled, making results unreadable. This was fixed in v0.2.0 by forcing the app's theme to light mode and disabling system force-dark at the window level.
- **Xiaomi/MIUI dark mode override (open issue):** on some Xiaomi/MIUI devices, the OS-level dark mode implementation may still override the app's forced light theme in certain cases, despite the v0.2.0 fix. A more targeted fix (applying `AppCompatDelegate.setDefaultNightMode()` at the Application level, before any activity is created) is planned for a future release.

---

# Current Limitations

Current limitations of the project include:

- Limited to ten disease classes
- Sensitive to poor lighting
- Dependent on image quality
- No lesion segmentation
- No multi-label prediction
- No severity estimation
- The input validation pipeline is heuristic/general-purpose-model based rather than trained specifically for this domain, and may occasionally over- or under-reject borderline images
- Educational use only

---

# Future Enhancements

Potential improvements include:

- EfficientNet-B0/B3
- MobileNetV3
- Vision Transformers (ViT)
- Transfer Learning
- Grad-CAM Explainability
- Lesion Segmentation
- Camera-based Real-time Detection
- Disease Information Module
- Treatment Recommendation System
- Cloud Synchronization
- Medical Report Generation
- Dedicated "not skin" negative class trained directly into the classifier, replacing the current heuristic-based validation pipeline
- MIUI-specific dark mode fix

---

# Disclaimer

This project has been developed solely for educational, research, and demonstration purposes.

The predictions generated by the application are based on patterns learned from the training dataset and should **not** be interpreted as medical advice or professional diagnosis.

Users should always consult qualified healthcare professionals for medical evaluation and treatment.