# Sentiment Analysis with Transformers

## Overview
This project implements a sentiment analysis model using transformer architecture, specifically designed to classify movie reviews from the IMDB dataset as positive or negative.

## Key Libraries
- **TensorFlow**: A powerful library for building and training machine learning models.
- **NumPy**: A fundamental package for numerical computations in Python.
- **Matplotlib**: A plotting library for creating static, animated, and interactive visualizations in Python.
- **Scikit-learn**: A machine learning library that provides simple and efficient tools for data mining and data analysis.
- **Seaborn**: A statistical data visualization library based on Matplotlib.
- **Gensim**: A library for topic modeling and document similarity analysis.
- **Keras**: An API for building and training deep learning models.

## Methodologies
1. **Data Preparation**: 
   - The IMDB reviews dataset is loaded and split into training, validation, and test sets.
   - A standardization function is defined to preprocess the text data by converting it to lowercase and removing HTML tags and punctuation.

2. **Text Vectorization**: 
   - The `TextVectorization` layer is used to convert raw text into integer sequences, which are then used as input to the model.

3. **Model Architecture**:
   - **Embeddings Layer**: Combines token embeddings with positional encodings to capture the order of words in the input sequences.
   - **Transformer Encoder**: Implements multi-head attention and feed-forward networks to process the input embeddings.
   - **Final Model**: The transformer model is built using the defined layers, and a sigmoid activation function is used for binary classification.

4. **Training**: 
   - The model is compiled with a binary cross-entropy loss function and trained using the Adam optimizer.
   - Training history is plotted to visualize loss and accuracy over epochs.

5. **Evaluation**: 
   - The model's performance is evaluated on the test dataset, and predictions can be made on new data.

## Conclusion
This project demonstrates the application of transformer models in natural language processing tasks, specifically sentiment analysis. The use of advanced techniques such as attention mechanisms allows for improved understanding and classification of text data.

