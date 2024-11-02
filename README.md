# Eye Direction Classification using Convolutional Neural Network (CNN)

This project focuses on classifying eye direction using CNN. It aims to predict the direction of gaze based on grayscale eye images, which are preprocessed to ensure consistent model training and evaluation.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements and Installation](#requirements-and-installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Running the Project](#running-the-project)
- [Future Improvements](#future-improvements)

## Overview

This project uses a deep learning model built with CNNs to classify images of eyes based on gaze direction. The model achieved a training accuracy of 97% and a validation accuracy of 95%, making it a reliable tool for eye direction classification tasks.

## Dataset

- **Source**: [Eye-dataset on Kaggle](https://www.kaggle.com/datasets/kayvanshah/eye-dataset)
- **Format**: Grayscale images focusing on the eyes.
- **Classes**: The dataset includes images categorized by various eye directions (e.g., forward, closed, left, right).
- **Structure**:
  - The images are labeled based on the direction of gaze.
  - The dataset is divided into training and validation sets for model evaluation.

## Requirements and Installation

Make sure to have the following libraries installed:

```bash
pip install kaggle tensorflow tensorflow_datasets keras matplotlib opencv-python
```

For running on Google Colab, upload the `kaggle.json` file to access Kaggle datasets. Place it in the root directory before running.

## Model Architecture

The model is designed using the Sequential API in Keras, with layers optimized for image recognition:

- **Input Layer**: Accepts grayscale eye images.
- **Convolutional Layers**: 3 convolutional layers with increasing filter sizes and ReLU activation.
- **Pooling Layers**: Max pooling layers to reduce feature map dimensionality.
- **Fully Connected Layers**: Dense layers for classification.
- **Output Layer**: Uses softmax activation for multi-class classification.

## Training

The model is trained using the following configuration:

- **Loss Function**: Categorical Crossentropy (suitable for multi-class classification).
- **Optimizer**: Adam optimizer for efficient gradient descent.
- **Epochs**: 50 epochs with early stopping.
- **Callbacks**: EarlyStopping and ModelCheckpoint to prevent overfitting.

## Evaluation

- **Training Accuracy**: 97%
- **Validation Accuracy**: 95%
- **Evaluation Metrics**: Accuracy and loss on the validation set to ensure the model generalizes well to unseen data.

## Running the Project

To reproduce the results, follow these steps:

1. Clone or download this repository.
2. Ensure `kaggle.json` is set up for Kaggle dataset access.
3. Download the dataset using the Kaggle API.
4. Run the preprocessing, model training, and evaluation cells in the notebook.

## Future Improvements

- **Data Augmentation**: Experiment with more aggressive augmentations.
- **Hyperparameter Tuning**: Adjust learning rates, batch sizes, and optimizer configurations.
- **Real-World Testing**: Test on diverse, real-world images to evaluate practical performance.

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)