# Intent Classification for Customer Service using Hugging Face Transformers

## Overview
This project focuses on building an **intent classification model** for customer service applications using **Hugging Face Transformers**. The model is designed to classify user intents based on their utterances, enabling automated responses in customer service scenarios.

## Installation
To set up the environment and install the necessary packages, run the following command:
```bash
pip install transformers datasets
```
This command installs the **Hugging Face Transformers** library, which provides pre-trained models and tools for natural language processing, as well as the **Datasets** library for easy access to various datasets.

## Imports
The following key libraries are used in the notebook:
```python
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizerFast
```

## Data Preparation
The dataset is prepared using the following steps:
1. **Kaggle Setup**: Install Kaggle and download the dataset using the Kaggle API.
2. **Load Dataset**: Use the `load_dataset` function to load the dataset from a CSV file.
3. **Preprocessing**: Map intents to numerical values for model training. The preprocessing function extracts utterances and their corresponding intents.

## Modeling
The model used is based on `TFBertForSequenceClassification`, which is a transformer model specifically designed for sequence classification tasks. The architecture leverages the BERT model's capabilities to understand context and semantics in text.

## Training
The model is trained for 2 epochs with the following code:
```python
history = model.fit(train_dataset, validation_data=val_dataset, epochs=2)
```
During training, the model learns to classify intents based on the provided utterances, optimizing its parameters to minimize the loss function.

## Evaluation
The evaluation is performed using a confusion matrix to visualize the model's performance. The confusion matrix provides insights into the model's accuracy and helps identify misclassifications.

## Testing
Example inputs can be tested with the model to predict intents:
```python
inputs = tokenizer(["Please how do I go about the account creation?", "After setting up my account, I feel like I need to change it. How do I go about that?", "How do I know how much I need to pay?", "Purchased a product, which I now want to change"], padding=True, return_tensors="tf")
logits = model(**inputs).logits
```
The model outputs the predicted intents for the provided utterances, demonstrating its functionality in a real-world scenario.

## Conclusion
This project showcases the application of Hugging Face Transformers for intent classification in customer service. The model can be further improved by fine-tuning hyperparameters, using larger datasets, or exploring different transformer architectures.
