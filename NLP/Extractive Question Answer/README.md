<h1 align="center">Extractive Question Answering Using Hugging Face Transformers</h1>

<p align="center">
  <strong>A deep learning project using Hugging Face 🤗 Transformers to extract answers from long passages with accuracy and efficiency.</strong>
</p>

<hr>

<h2>🚀 Project Overview</h2>
<p>
  This project implements an <strong>Extractive Question Answering (QA)</strong> system using Hugging Face's Longformer model. The system is fine-tuned on the TriviaQA dataset to enable accurate extraction of answers to questions from long contextual paragraphs.
</p>

<h2>🛠️ Key Features</h2>
<ul>
  <li><strong>Preprocessing and Tokenization</strong>: Efficient handling of input data, ensuring compatibility with the Longformer model.</li>
  <li><strong>Fine-tuning</strong>: Training the model on the TriviaQA dataset to improve its accuracy and performance.</li>
  <li><strong>Answer Extraction</strong>: Extracting precise answers using start and end logits from the model's output.</li>
  <li><strong>Evaluation</strong>: Assessed performance using SQuAD metrics to ensure real-world reliability.</li>
</ul>

<hr>

<h2>🧩 Installation and Dependencies</h2>
<ol>
  <li>Install required libraries:</li>
</ol>

<pre>
<code>pip install transformers datasets evaluate</code>
</pre>

<p>Additional imports used in this project include:</p>
<ul>
  <li>TensorFlow</li>
  <li>NumPy</li>
  <li>Matplotlib</li>
  <li>Sklearn</li>
</ul>

<hr>

<h2>📂 Dataset Used</h2>
<p><strong>TriviaQA</strong>: A large-scale dataset for QA tasks.</p>
<pre>
<code>
from datasets import load_dataset
dataset = load_dataset("trivia_qa")
</code>
</pre>

<hr>

<h2>⚙️ How It Works</h2>
<ol>
  <li><strong>Preprocessing:</strong> Tokenizes questions and passages using the Hugging Face tokenizer, ensuring proper alignment of answer positions with token indices.</li>
  <li><strong>Fine-tuning:</strong> The Longformer model is fine-tuned on the TriviaQA dataset using TensorFlow.</li>
  <li><strong>Inference:</strong> Answers are predicted by identifying the start and end positions from the model's output logits:</li>
</ol>

<pre>
<code>
answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens)
</code>
</pre>

<ol start="4">
  <li><strong>Evaluation:</strong> The model's predictions are evaluated using SQuAD metrics for exact match and F1 scores.</li>
</ol>

<hr>

<h2>📊 Results</h2>
<ul>
  <li>The fine-tuned model achieved reliable performance with high accuracy in extracting answers from long paragraphs.</li>
  <li>Evaluation metrics, such as F1 score and exact match, demonstrate the effectiveness of the system.</li>
</ul>

<hr>

<h2>🛠️ Usage</h2>
<ol>
  <li>Clone the repository:</li>
</ol>

<pre>
<code>
git clone &lt;https://github.com/SangamNirala/Data_Science/tree/main/NLP/Extractive%20Question%20Answer&gt;
cd &lt;repository-folder&gt;
</code>
</pre>

<ol start="2">
  <li>Run the model on your input:
    <ul>
      <li>Provide a question and context as input.</li>
      <li>The system extracts the most probable answer using the Longformer model.</li>
    </ul>
  </li>
</ol>

<hr>

<h2>🤝 Contributions</h2>
<p>
  Contributions, issues, and feature requests are welcome! Feel free to open a pull request or raise an issue on GitHub.
</p>

<hr>

<h2>📜 License</h2>
<p>
  This project is licensed under the <a href="LICENSE">MIT License</a>.
</p>

<hr>

<p align="center">
  Made with ❤️ using Hugging Face Transformers
</p>
