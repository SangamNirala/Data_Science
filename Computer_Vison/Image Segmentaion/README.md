# Image Segmentation Project

This project implements an image segmentation model using TensorFlow and Keras. The model is designed to segment images into different classes based on the input data. The architecture leverages a pre-trained ResNet50 model as the backbone for feature extraction, followed by custom layers for upsampling and classification.

The project utilizes several key libraries including **TensorFlow**, **NumPy**, **Matplotlib**, **OpenCV**, **Albumentations**, and **TensorFlow Datasets**. 

### Project Overview

#### Data Preparation
The project begins with data preparation, where images and their corresponding segmentation maps are loaded. A custom `DataGenerator` class is implemented to handle the loading and preprocessing of the data in batches.

#### Modeling
The model architecture is built using TensorFlow's Keras API. A base model (ResNet50) is used for feature extraction, and custom layers such as `Upsample` and `ConvLayers` are added for upsampling and classification.

#### Training
The model is compiled with the **Adam** optimizer and **Sparse Categorical Crossentropy** loss function. A callback function, **ModelCheckpoint**, is used to save the best model weights during training.

#### Testing
After training, the model is tested on new images, and the segmentation results are visualized using **Matplotlib**.

### Conclusion
This project demonstrates the application of deep learning techniques for image segmentation tasks, showcasing the use of various libraries and custom implementations to achieve the desired results.
