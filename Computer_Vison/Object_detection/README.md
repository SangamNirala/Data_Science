# YOLO Object Detection from Scratch

## Description
This project implements the YOLO (You Only Look Once) object detection algorithm from scratch using TensorFlow. YOLO is a state-of-the-art, real-time object detection system that can detect multiple objects in images and video streams. This implementation utilizes the Pascal VOC 2012 dataset for training and validation.

## Installation
To set up the project, follow these steps:
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The project uses the Pascal VOC 2012 dataset, which can be downloaded using the Kaggle API. Ensure you have your Kaggle API credentials set up. The dataset includes images and annotations for various object classes such as aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dining table, dog, horse, motorbike, person, potted plant, sheep, sofa, train, and TV monitor.

## Data Preparation
The dataset is prepared by:
- Downloading the dataset using Kaggle API.
- Splitting the dataset into training and validation sets.
- Preprocessing the annotations to extract bounding boxes and class labels.

## Usage
To run the object detection model, use the following command:
```bash
python detect.py --image <image-path>
```
Replace `<image-path>` with the path to the image you want to analyze.

## Training
The model is trained using the following command:
```python
history = model.fit(train_dataset, validation_data=val_dataset, epochs=135, callbacks=[lr_callback, callback])
```
The model architecture is based on EfficientNetB1, and it includes several convolutional layers, batch normalization, and dropout for regularization.

## Testing
To test the model on images, use the following function:
```python
model_test(filename)
```
Replace `filename` with the name of the image file you want to test. The model will output the detected objects with bounding boxes.

## Testing Process
- The model was tested using the `model_test` function.
- The function loads a trained YOLO model and processes images to detect objects.
- It reads images from a specified directory and resizes them.
- The model predicts object positions based on the input images.
- A threshold of 0.25 is applied to filter predictions, ensuring only confident detections are considered.
- Non-maximum suppression is used to eliminate overlapping bounding boxes.
- The final output includes a list of detected objects with their respective bounding boxes and confidence scores.

## Loss Function
The YOLO loss function is implemented to calculate the loss based on object presence, bounding box coordinates, and class predictions. It includes components for object loss, no-object loss, class loss, and bounding box loss.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License.

## Acknowledgments
- TensorFlow
- OpenCV
- Kaggle
- Pascal VOC Dataset
