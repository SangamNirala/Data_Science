# People Counting by Neuralearn AI

## Overview
This project implements a people counting system using deep learning techniques. The model is built on TensorFlow and utilizes a convolutional neural network (CNN) architecture to predict the number of people in images.

## Dependencies
The following libraries are required to run this project:

- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV
- SciPy

## Key Components
- **Data Preparation**: The data is prepared by loading images and their corresponding density maps. A Gaussian distribution is used to create density maps based on the ground truth points.
- **Modeling**: The model is built using the VGG16 architecture as a base, with additional convolutional layers added for feature extraction. The model is compiled with the Sparse Categorical Crossentropy loss function and the Adam optimizer.
- **Training**: The model is trained using a custom data generator that yields batches of images and their corresponding density maps.
- **Testing**: The model's performance is evaluated by predicting the number of people in test images and comparing the predictions with ground truth values.

## Usage
1. Install the required dependencies.
2. Prepare your dataset of images and corresponding density maps.
3. Run the training script to train the model.
4. Use the trained model to predict the number of people in new images.

## License
This project is licensed under the MIT License.
