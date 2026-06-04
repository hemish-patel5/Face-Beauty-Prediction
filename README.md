# Face Beauty Prediction 

This project implements a deep learning-based approach to predict facial beauty scores from photographs using the SCUT-FBP5500 dataset.

## Project Overview
Facial beauty assessment is a complex computer vision task that involves analyzing facial features, symmetry, and proportions. This project leverages Convolutional Neural Networks (CNNs) and transfer learning to predict subjective beauty scores with high accuracy.


## Abstract
In this project, we propose a deep learning-based approach for automatic facial beauty prediction from photographs. Facial beauty assessment is a challenging computer vision task due to the subjective nature of beauty perception and the complex interplay of facial features, proportions, and symmetry. We develop a convolutional neural network (CNN) architecture based on fine-tuned pre-trained models (ResNet50 and VGG16) to predict beauty scores from facial images. Our methodology includes comprehensive data preprocessing, face alignment, augmentation techniques, and transfer learning to address the limited availability of labeled beauty datasets. We evaluate our models on the SCUT-FBP5500 benchmark dataset, which contains 5,500 frontal facial images with corresponding beauty scores. Experimental results demonstrate that our fine-tuned ResNet50 model achieves a Pearson correlation coefficient of 0.89 and mean absolute error of 0.32 on a 5-point scale, outperforming baseline approaches. This research contributes to understanding how deep learning models can capture aesthetic preferences and has applications in cosmetic recommendations, plastic surgery planning, and social media analytics.

## Dataset
- **Name:** [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database)
- **Description:** A benchmark dataset containing 5,500 frontal facial images with diverse beauty scores.

## Methodology
1. **Preprocessing:** Comprehensive data cleaning, face alignment, and normalization.
2. **Augmentation:** Application of various augmentation techniques to improve model generalization.
3. **Transfer Learning:** Utilization of pre-trained models (ResNet50 and VGG16) fine-tuned for the regression task of beauty scoring.
4. **Evaluation:** Performance measured using Mean Absolute Error (MAE) and Pearson Correlation Coefficient (PCC).

## Results
Our fine-tuned ResNet50 model demonstrated superior performance:
- **Pearson Correlation Coefficient (PCC):** 0.89
- **Mean Absolute Error (MAE):** 0.32 (on a 5-point scale)

## Applications
- Cosmetic recommendations
- Plastic surgery planning
- Social media analytics
