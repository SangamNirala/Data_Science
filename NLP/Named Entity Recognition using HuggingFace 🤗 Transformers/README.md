# Named Entity Recognition using Hugging Face 🤗 Transformers

## Description
This project implements **Named Entity Recognition (NER)** using **Hugging Face's Transformers** library. It utilizes the **CoNLL-2003 dataset** to train a model that can identify and classify named entities in text. The project leverages **TensorFlow** for model training and evaluation, employing techniques such as tokenization and data collation to prepare the dataset effectively.

## Installation
To install the required libraries, run the following commands:
```bash
pip install transformers datasets evaluate
pip install seqeval
```

## Libraries Used
- **TensorFlow**: For building and training the model.
- **NumPy**: For numerical computations.
- **Matplotlib**: For plotting and visualizations.
- **scikit-learn**: For machine learning utilities.
- **OpenCV**: For image processing tasks.
- **Seaborn**: For enhanced visualizations.
- **Gensim**: For word embeddings and related tasks.
- **Hugging Face Transformers**: For pre-trained models and tokenization.

## Data Preparation
The project uses the **CoNLL-2003 dataset**, which is loaded and tokenized for training. The labels are aligned with the tokens for proper training, ensuring that the model can learn to predict named entities accurately.

## Model Training
The model is trained using the **TFRobertaForTokenClassification** architecture from Hugging Face. The training process includes defining the optimizer and compiling the model, allowing it to learn from the prepared dataset.

## Evaluation
The model's performance is evaluated using the **seqeval** metric, which computes precision, recall, and F1 score for the NER task, providing insights into the model's effectiveness.

## Testing
The model is tested with sample inputs to demonstrate its ability to predict named entities, showcasing its practical application in real-world scenarios.
