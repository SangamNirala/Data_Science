<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Named Entity Recognition (NER) using HuggingFace Transformers</title>
</head>
<body>

<h1>Named Entity Recognition (NER) using HuggingFace 🤗 Transformers</h1>

<p>This project demonstrates how to perform Named Entity Recognition (NER) using the HuggingFace 🤗 Transformers library. The NER model is based on RoBERTa, a transformer model pre-trained on large amounts of text and fine-tuned for token classification tasks.</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#imports">Imports</a></li>
    <li><a href="#data-preparation">Data Preparation</a></li>
    <li><a href="#modeling">Modeling</a></li>
    <li><a href="#training">Training</a></li>
    <li><a href="#evaluation">Evaluation</a></li>
    <li><a href="#testing">Testing</a></li>
</ul>

<h2 id="installation">Installation</h2>

<p>To get started, you need to install the necessary Python libraries. Run the following commands in your environment:</p>

<pre><code>pip install transformers datasets evaluate
pip install seqeval</code></pre>

<h2 id="imports">Imports</h2>

<p>The following libraries are imported for various functionalities like data loading, model building, and evaluation:</p>

<pre><code>import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import cv2
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import datetime
import pathlib
import io
import os
import re
import string
import evaluate
import time
import gensim.downloader as api
from PIL import Image
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from transformers import (
    BertTokenizerFast, TFBertTokenizer, BertTokenizer,
    RobertaTokenizerFast, DataCollatorForTokenClassification,
    TFRobertaForSequenceClassification, TFBertForSequenceClassification,
    TFBertModel, create_optimizer, TFRobertaForTokenClassification,
    TFAutoModelForTokenClassification
)</code></pre>

<h2 id="data-preparation">Data Preparation</h2>

<p>We use the <code>conll2003</code> dataset for training and evaluating the NER model. The dataset is loaded using HuggingFace’s <code>datasets</code> library.</p>

<pre><code>dataset = load_dataset("conll2003")</code></pre>

<p>We then tokenize the dataset using the RoBERTa tokenizer. Labels are aligned with tokens to ensure consistency during training.</p>

<pre><code>tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            continue
    return new_labels

def tokenizer_function(dataset):
    out = tokenizer(dataset["tokens"], truncation=True, is_split_into_words=True)
    out['labels'] = align_labels_with_tokens(dataset["ner_tags"], out.word_ids())
    return out

tokenized_dataset = dataset.map(tokenizer_function, remove_columns=['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])</code></pre>

<h2 id="modeling">Modeling</h2>

<p>The model is based on <code>TFRobertaForTokenClassification</code>, which is suitable for token-level classification tasks like NER. The model is initialized using the pre-trained RoBERTa model.</p>

<pre><code>model = TFRobertaForTokenClassification.from_pretrained(
    "roberta-base",
    num_labels=9
)
model.summary()</code></pre>

<h2 id="training">Training</h2>

<p>We train the model using TensorFlow and the tokenized dataset. The model is compiled with the Adam optimizer and evaluated using the sequence labeling task.</p>

<pre><code>optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

model.compile(optimizer=optimizer)

history = model.fit(
    tf_train_dataset,
    validation_data=tf_val_dataset,
    epochs=NUM_EPOCHS
)</code></pre>

<p>The training progress can be monitored using loss curves:</p>

<pre><code>plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()</code></pre>

<h2 id="evaluation">Evaluation</h2>

<p>After training the model, the evaluation is performed using the <code>seqeval</code> metric, which is designed specifically for evaluating sequence labeling tasks.</p>

<pre><code>metric = evaluate.load("seqeval")
ind_to_label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
all_predictions = []
all_labels = []

for batch in tf_val_dataset:
    logits = model.predict(batch)["logits"]
    labels = batch["labels"].numpy()
    predictions = tf.argmax(logits, axis=-1).numpy()
    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(ind_to_label[predicted_idx])
            all_labels.append(ind_to_label[label_idx])

print(metric.compute(predictions=[all_predictions], references=[all_labels]))</code></pre>

<h2 id="testing">Testing</h2>

<p>For testing, you can pass a sentence and get predictions for the named entities. The sentence "Wake Up Joe Marshal, you just got a call from UNESCO for a trip to India" can be tested as follows:</p>

<pre><code>inputs = tokenizer(["Wake Up Joe Marshal, you just got a call from UNESCO for a trip to India"], padding=True, return_tensors="tf")
logits = model(**inputs).logits
predictions = tf.argmax(logits, axis=-1)

out_str = ""
for i in range(1, len(inputs.tokens()) - 1):
    if predictions[0][i] != 0:
        out_str += f" {inputs.tokens()[i]} ---> {ind_to_label[predictions[0][i].numpy()]}"
    else:
        out_str += f" {inputs.tokens()[i]}"

print(out_str.replace("Ġ", ""))</code></pre>

<p>This will output the entities recognized in the sentence.</p>

<p>Feel free to contribute or open issues for improvements or bug reports. Happy coding! 😊</p>

</body>
</html>
