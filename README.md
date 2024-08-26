# Potato Disease Classification using CNN

A brief description of what this project does and who it's for

## Table of Contents

1. [Overview](#overview)
2. [User Interface](#user-interface)
3. [Problem Statement](#problem-statement)
4. [Dataset](#dataset)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Methodology](#methodology)
8. [Implementation](#implementation)
9. [Testing](#testing)
10. [Result and Prediction](#result-and-prediction)
11. [Conclusion](#conclusion)
12. [References](#references)

## Overview

This project involves the development of a convolutional neural network (CNN) to classify potato plant diseases into three categories: Early Blight, Late Blight, and Healthy. The model aims to assist farmers and agronomists in early detection of these diseases through image classification, potentially improving crop management and yield.

## Problem Statement

Potato crops are prone to various diseases that can significantly affect yield and quality. Early and accurate identification of these diseases is crucial for effective management. Traditional methods of disease detection are often labor-intensive and subjective. This project seeks to automate the detection process using a CNN model trained on labeled images of potato plants, thereby improving accuracy and efficiency.

## Dataset

For dataset check Kaggle --> [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)

The dataset used for training and testing the model consists of 2,152 images categorized into three classes: `Potato___Early_blight`,`Potato___Late_blight`, and `Potato___Healthy`. The images are pre-processed to a uniform size of 256x256 pixels, with data augmentation applied to enhance model generalization.

## Installation

Follow these steps to set up and run the project on your local machine.

### Prerequisites

Make sure you have the following installed:

- **Python 3.x**
- **pip** (Python package installer)
- **TensorFlow**
- **Juypter Notebook**
- **Flask**

### Steps

1. **Clone the Repository:**

   Clone this repository to your local machine:

   ```bash
   git clone https://github.com/charang2003/potato-disease-classification.git
   cd potato-disease-classification

   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To run the webpage of application:

```bash
python app.py
```

## Methodology

1. **Data Preprocessing**

- Image resizing to 256x256 pixels.
- Data augmentation including random flips and rotations.
- Normalization of pixel values.

2. **Model Architecture**

- A sequential CNN model with six convolutional layers, each followed by a max-pooling layer.
- Fully connected dense layers with ReLU activation.
- Softmax output layer for multi-class classification.

3. **Training and Evaluation**

- The model was trained for 50 epochs using the Adam optimizer and Sparse Categorical Crossentropy loss.
- Performance metrics include accuracy and loss, evaluated on both the training and validation datasets.

## Implementation

1. **Data Preparation**

- Load and split the dataset into training, validation, and testing sets.
- Apply data augmentation and normalization.

2. **Model Building**

- Design a CNN architecture with multiple convolutional layers.
- Compile the model with appropriate loss function and optimizer.
- Train the model using the training dataset and validate it using the validation dataset.

3. **Model Evaluation**

- Evaluate the model on the test dataset to determine accuracy and loss.
- Save the trained model for future predictions.

4. **Deployment**

- Deploy the model using Flask for real-time predictions.

## Testing

The model was tested on a separate test dataset to evaluate its performance. It achieved an accuracy of 100% on the test set, demonstrating its effectiveness in classifying the potato diseases. The prediction confidence for correctly classified images was consistently high, confirming the reliability of the model.

## Result and Prediction

The model was tested on a separate dataset to evaluate its performance in classifying potato leaf diseases. Below are examples of different predictions made by the model, along with the corresponding images:

### Example predictions

1. **Prediction**: Early Blight

- **Confidence**: 95.4%
- **Image**:

2. **Prediction**: Late Blight

- **Confidence**: 95.4%
- **Image**:

3. **Prediction**: Healthy

- **Confidence**: 95.4%
- **Image**:

### Model Performance Metrics

The model achieved an overall accuracy of approximately 98% on the test dataset. The following metrics were recorded during evaluation:

- **Validation Accuracy**: 96.74%
- **Loss**: 0.0226

The model demonstrates strong performance in distinguishing between the three classes. However, occasional misclassifications suggest potential areas for further improvement, such as fine-tuning hyperparameters or expanding the training dataset.

## Conclusion

- The CNN model developed in this project effectively classifies potato plant diseases with high accuracy.
- The model's robustness was validated through extensive testing, where it demonstrated strong performance across various scenarios.
- The project's success highlights the potential of deep learning in automating agricultural diagnostics, paving the way for further advancements in smart farming technologies.

## References

1. TensorFlow Documentation: https://www.tensorflow.org/
2. Kaggle Potato Disease Dataset: https://www.kaggle.com/datasets/vipoooool/potato-disease
3. Deep Learning with Python by François Chollet, Manning Publications, 2017.
