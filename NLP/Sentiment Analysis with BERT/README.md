# README for Sentiment Analysis with BERT

## Project Overview
This project implements a sentiment analysis model using BERT and other transformer architectures. It leverages the Hugging Face Transformers library to classify movie reviews from the IMDB dataset.

## Installation
To set up the project, install the required libraries using pip:
```bash
pip install transformers datasets tensorflow matplotlib scikit-learn opencv-python seaborn gensim
```

## Project Structure
- **Data Preparation**: The project loads and preprocesses the IMDB dataset for training and validation.
- **Modeling**: It includes implementations for BERT and RoBERTa models for sequence classification.
- **Training**: The model is trained using TensorFlow, with metrics for loss and accuracy tracked.
- **Testing**: The trained model is evaluated on test data, and predictions are made on sample inputs.
- **Conversion to ONNX**: The model is converted to ONNX format for optimized inference.
- **Quantization**: The ONNX model is quantized to reduce its size and improve inference speed.

## Usage
1. **Data Preparation**: The dataset is loaded and tokenized using BERT and RoBERTa tokenizers.
2. **Model Training**: The model is trained for a specified number of epochs, and training history is plotted.
3. **Model Evaluation**: After training, the model's performance is evaluated on the validation set.
4. **Inference**: The model can be used to predict sentiments of new movie reviews.
5. **ONNX Conversion**: The trained model can be exported to ONNX format for deployment.

## Key Components
- **Imports**: The project uses various libraries such as TensorFlow, NumPy, and Hugging Face Transformers.
- **Modeling**: Different models are implemented, including TFBertForSequenceClassification and TFRobertaForSequenceClassification.
- **Training and Evaluation**: The model is trained with a binary cross-entropy loss function and evaluated using accuracy metrics.
- **Visualization**: Training and validation loss and accuracy are plotted for analysis.

## Conclusion
This project demonstrates the application of BERT for sentiment analysis, showcasing the power of transformer models in natural language processing tasks.
