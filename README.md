# Potato Disease Classification using CNN

This project focuses on the classification of potato plant diseases using a Convolutional Neural Network (CNN) model implemented in TensorFlow. The primary goal is to accurately identify early blight and late blight diseases from images of potato leaves.

## Table of Contents
- [Features](#features)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Preprocessing**: Techniques for preparing the dataset, including resizing images, normalization, and augmentation.
- **Model Architecture**: A CNN model designed to extract relevant features and classify images into healthy, early blight, or late blight categories.
- **Training and Evaluation**: Training the model with labeled data and evaluating its performance using metrics such as accuracy, precision, recall, and F1-score.
- **Visualization**: Use of Matplotlib to visualize the training process and model predictions.

## Technologies

- TensorFlow for building and training the CNN model.
- Matplotlib for data visualization and analysis.
- Python for scripting and implementation.

## Dataset

The dataset consists of images of potato leaves categorized into three classes: healthy, early blight, and late blight. It can be downloaded from [Kaggle](https://www.kaggle.com/). Make sure to place the dataset in the `data/` directory.

## Model Architecture

The CNN model architecture includes multiple convolutional layers followed by max-pooling layers, dropout layers to prevent overfitting, and dense layers for classification. The architecture is designed to efficiently extract and learn features from the input images.

## Training

The model is trained using the Adam optimizer and categorical cross-entropy loss function. Training involves feeding the preprocessed images into the model and adjusting the weights to minimize the loss function.

## Evaluation

The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. A confusion matrix is also generated to visualize the classification results.

## Visualization

Matplotlib is used to visualize the training process, including loss and accuracy curves. Additionally, sample predictions and corresponding ground truth labels are displayed to demonstrate the model's performance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/potato-disease-classification.git
   cd potato-disease-classification
