# Oral Cancer Detection using Machine Learning and Deep Learning Models

# Introduction
This document outlines the workflow and methodology for developing an automated **oral cancer detection system** using histopathological images.  
The study compares the performance of multiple **deep learning** and **traditional machine learning** models to accurately classify oral cancer at an early stage — a critical factor in improving patient survival rates (above 80% if detected early).  

The following sections describe the dataset preprocessing, model training process, transfer learning approaches, and evaluation of different classifiers including CNN, EfficientNetB0, DenseNet121, SVC, Random Forest, and Decision Tree.  

# Project Setup

```python
# Load necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, DenseNet121
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



## 1. Data Preparation

Load and preprocess the histopathological image dataset used for oral cancer detection.

- **Image Loading:** Load labeled histopathological images (cancerous vs. non-cancerous).  
- **Image Preprocessing:** Resize, normalize pixel values, and convert to arrays.  
- **Data Augmentation:** Apply rotation, flipping, zooming, and brightness adjustment to improve generalization.  
- **Train-Test Split:** Divide dataset into training, validation, and test sets for fair model evaluation.  

---

## 2. Feature Extraction

The initial experiments with traditional ML models (SVC, Random Forest, Decision Tree) used extracted image features such as texture, color histograms, and edge features.  
However, these models achieved only **50–75% accuracy**, indicating difficulty in capturing complex visual patterns from medical images.

To address this limitation, **Convolutional Neural Networks (CNNs)** and **transfer learning** were employed to extract deep hierarchical features.

---

## 3. Deep Learning and Transfer Learning Models

Multiple architectures were trained and evaluated for performance:

### **Basic CNN**
A simple CNN model was built from scratch using convolutional, pooling, and dense layers.  
However, due to limited dataset size and feature complexity, the model underperformed.

### **EfficientNetB0 (Pre-trained)**
Used transfer learning by fine-tuning the EfficientNetB0 model pre-trained on ImageNet.  
- Achieved the **highest accuracy of 92%**.  
- Effectively captured high-level spatial and textural features from histopathological images.

### **DenseNet121 (Pre-trained)**
Applied a similar transfer learning approach with DenseNet121.  
- Achieved **89% accuracy**, slightly below EfficientNetB0.  
- Demonstrated strong feature reuse and connectivity benefits.

---

## 4. Conventional Machine Learning Models

Traditional classifiers were also implemented for comparison:

- **Support Vector Classifier (SVC):** Achieved moderate accuracy but limited feature extraction capability.  
- **Random Forest:** Provided stable performance but lacked ability to learn spatial dependencies.  
- **Decision Tree:** Simple model with interpretability but poor accuracy (~50–60%).  

These results highlight the superiority of **deep learning** and **transfer learning** for image-based cancer detection.

---

## 5. Model Evaluation and Conclusion

Each model was evaluated using metrics such as **accuracy**, **confusion matrix**, and **classification report**.  

| Model | Accuracy (%) |
|--------|---------------|
| EfficientNetB0 | **92** |
| DenseNet121 | 89 |
| Basic CNN | 70 |
| Random Forest | 68 |
| SVC | 72 |
| Decision Tree | 55 |

### **Conclusion**

Transfer learning models — particularly **EfficientNetB0** — demonstrated outstanding performance in identifying oral cancer from complex histopathological images.  
This validates the effectiveness of leveraging pre-trained features for medical image analysis.

Future work should focus on:
- Enhancing dataset quality with multiple cancer stages and larger sample sizes.  
- Improving image preprocessing and augmentation pipelines.  
- Integrating model deployment pipelines for clinical diagnostics where pathologists can upload tissue images and receive automated results — reducing human error and improving early detection rates.  

---

## References

- Tan, M. & Le, Q. V. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, Google AI, 2019.  
- Huang, G. et al. *Densely Connected Convolutional Networks*, CVPR, 2017.  
- *Oral Cancer Research Statistics – World Health Organization (WHO)*, 2023.  

---

## Appendix

This appendix includes general Python code snippets for data preprocessing, model definition, and evaluation.  
Refer to the supplementary notebook or report for detailed hyperparameters, augmentation strategies, and fine-tuning procedures.
