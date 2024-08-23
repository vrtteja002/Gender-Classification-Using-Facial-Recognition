# Gender Classification using VGG16

## Overview
This project implements a deep learning model for gender classification using a pre-trained VGG16 network. The model is fine-tuned on a custom dataset to classify images as either "Male" or "Female".

## Features
- Utilizes transfer learning with a pre-trained VGG16 model
- Implements data augmentation techniques for improved generalization
- Achieves high accuracy in gender classification

## Requirements
- Python 3.x
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- numpy
- PIL

## Dataset
The dataset is organized into three subsets:
- Training set: `/kaggle/input/gender-classification/Gender Dataset/train`
- Validation set: `/kaggle/input/gender-classification/Gender Dataset/valid`
- Test set: `/kaggle/input/gender-classification/Gender Dataset/test`

Each subset contains images categorized into "Male" and "Female" classes.

## Setup and Installation
1. Clone this repository:
   ```
   git clone https://github.com/your-username/gender-classification-vgg16.git
   cd gender-classification-vgg16
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Organize your images into train, valid, and test directories
   - Ensure each directory has subdirectories for "Male" and "Female" classes

## Usage
1. To train the model:
   ```
   python train.py
   ```

2. To evaluate the model on the test set:
   ```
   python evaluate.py
   ```

## Model Architecture
The project uses a pre-trained VGG16 model with the following modifications:
- The final fully connected layer is replaced with a new layer for binary classification
- Only the classifier part of the model is fine-tuned, while the feature extractor remains frozen

## Data Augmentation
The following augmentation techniques are applied to the training data:
- Random horizontal flip (50% probability)
- Random vertical flip (10% probability)
- Random grayscale conversion (10% probability)
- Gaussian blur

## Results
After training for 60 epochs, the model achieves:
- Validation Accuracy: 90.70%
- Test Accuracy: 95.54%

Detailed classification reports for both validation and test sets are provided in the output.

## Visualizations
The training script generates plots for:
- Training and Validation Loss
- Training and Validation Accuracy

These plots help in monitoring the model's performance and identifying potential overfitting.

## Future Improvements
- Experiment with other pre-trained models (e.g., ResNet, EfficientNet)
- Implement cross-validation for more robust evaluation
- Explore additional data augmentation techniques
- Fine-tune hyperparameters for potentially better performance

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.
