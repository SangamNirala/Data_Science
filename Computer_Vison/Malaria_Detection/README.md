# Malaria Detection Project

## Overview
This project implements a machine learning model for detecting malaria in cell images using deep learning techniques with TensorFlow and Keras. The model is trained on a dataset of cell images, which includes both parasitized and uninfected cells, aiming to assist in the early diagnosis of malaria.

## Technologies Used
- **TensorFlow**: A powerful library for building and training deep learning models.
- **Keras**: A high-level API for TensorFlow that simplifies the process of model building and training.
- **NumPy**: For efficient numerical computations and array manipulations.
- **Matplotlib**: For data visualization, including plotting training history and confusion matrices.
- **Scikit-learn**: For machine learning utilities, including metrics for model evaluation.
- **OpenCV**: For image processing tasks, such as image resizing and augmentation.
- **WandB (Weights & Biases)**: For experiment tracking, model versioning, and hyperparameter tuning.

## Dataset
The dataset used in this project is the Malaria dataset from TensorFlow Datasets, which contains a total of 27,558 cell images with equal instances of parasitized and uninfected cells. The dataset is split into training, validation, and test sets to evaluate the model's performance effectively.

## Data Preprocessing
Data preprocessing techniques applied in this project include:
- **Data Augmentation**: Techniques such as rotation, flipping, and brightness adjustment are used to increase the diversity of the training dataset and improve model generalization.
- **Resizing and Rescaling**: Images are resized to a uniform size (224x224 pixels) and rescaled to a range of [0, 1] to normalize the input data.

## Model Architecture
The model is built using a Convolutional Neural Network (CNN) architecture, which includes:
- **Convolutional Layers**: For feature extraction from the input images.
- **MaxPooling Layers**: For down-sampling the feature maps and reducing dimensionality.
- **Dropout Layers**: To prevent overfitting by randomly dropping units during training.
- **Dense Layers**: For classification, with a final output layer using a sigmoid activation function for binary classification.

## Training
The model is trained using the following hyperparameters:
- **Learning Rate**: 0.001
- **Batch Size**: 128
- **Number of Epochs**: 5
- **Dropout Rate**: 0.0
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam optimizer for efficient training.

## Evaluation
The model's performance is evaluated using various metrics, including:
- **Accuracy**: The proportion of correctly classified instances.
- **Precision**: The ratio of true positive predictions to the total predicted positives.
- **Recall**: The ratio of true positive predictions to the total actual positives.
- **AUC (Area Under the Curve)**: A performance measurement for classification problems at various threshold settings.
- **Confusion Matrix**: A table used to describe the performance of the model on the test dataset.



## Conclusion
This project demonstrates the application of deep learning techniques for image classification tasks, specifically in the medical field for malaria detection. The model's ability to accurately classify cell images can significantly aid healthcare professionals in diagnosing malaria at an early stage, potentially saving lives.

## Future Work
Future enhancements may include:
- Implementing more advanced architectures such as transfer learning with pre-trained models.
- Exploring additional data augmentation techniques to further improve model robustness.
- Conducting hyperparameter tuning to optimize model performance.
