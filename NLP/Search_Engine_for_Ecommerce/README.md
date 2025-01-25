# E-commerce Search Engine using Sentence Transformers

## Overview
This project implements a search engine for **e-commerce** products using **Sentence Transformers** from **Hugging Face**. The model is designed to enhance product search capabilities by understanding the semantic similarity between user queries and product descriptions.

## Requirements
- Python 3.x
- TensorFlow
- Transformers
- Datasets
- Other libraries as specified in the notebook

## Installation
To install the required libraries, run:
```bash
pip install transformers datasets
```

## Dataset Preparation
1. **Download Datasets**: The datasets are downloaded using the AICrowd CLI.
   ```bash
   !pip install aicrowd-cli
   !aicrowd login
   !aicrowd dataset download -c esci-challenge-for-improving-product-search
   ```
2. **Data Loading**: The datasets are loaded into pandas DataFrames for processing.

## Data Preprocessing
- The data is preprocessed by tokenizing the queries and products using the **AutoTokenizer** from Hugging Face.
- Labels are mapped to numerical values for model training, which is crucial for effective **model training**.

## Model Definition
A custom Keras model is defined using a pre-trained transformer model. The model includes:
- Input layers for queries and products.
- A dense layer for output.
- Mean pooling to aggregate token embeddings, which is essential for calculating **cosine similarity**.

## Model Training
The model is trained on the prepared dataset for a specified number of epochs using the **Adam** optimizer and **Binary Crossentropy** loss function. This training process is vital for achieving high performance in product recommendations.

## Testing
- The model creates embeddings for product titles and computes cosine similarity for recommendations.
- The cosine similarity is used to find the most relevant products based on user queries, enhancing the overall search experience.

## Usage
To use the model, load the trained weights and input your queries to get product recommendations based on semantic similarity.

## License
This project is licensed under the MIT License.
