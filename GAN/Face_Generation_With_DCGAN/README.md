# DCGAN Image Generation

## Project Overview
This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using TensorFlow to generate images from the CelebA dataset. The model consists of a generator and a discriminator, which are trained together in a competitive manner.

## Key Features
- **Data Preparation**: Downloads and prepares the CelebA dataset for training.
- **Model Architecture**: Defines the generator and discriminator models using Keras.
- **Training**: Trains the GAN for a specified number of epochs and saves generated images at the end of each epoch.
- **Visualization**: Plots the loss of the generator and discriminator over epochs.

## Installation
To run this project, ensure you have the following dependencies installed:
- TensorFlow
- NumPy
- Matplotlib
- Kaggle (for dataset access)

## Usage
1. Clone the repository or download the Jupyter Notebook.
2. Install the required packages:
   ```bash
   pip install tensorflow numpy matplotlib kaggle
   ```
3. Set up your Kaggle API credentials by placing the `kaggle.json` file in the appropriate directory.
4. Run the Jupyter Notebook to train the GAN and generate images.

## Example Code Snippet
```python
# Example of training the GAN
gan.fit(train_dataset, epochs=EPOCHS, callbacks=[ShowImage(LATENT_DIM)])
```

## License
This project is licensed under the MIT License.

## Author
Neuralearn.ai
