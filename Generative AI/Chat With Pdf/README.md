# PDF Chat Application

## Overview
The PDF Chat Application is a **Streamlit**-based tool designed for **interactive document processing**. It allows users to engage with **PDF documents** through a chat interface, adapting its responses based on the **document type** and the **usage context**. This application leverages advanced **natural language processing** techniques to provide insightful responses tailored to various document types.

## Features
- **Document Type Specialization**: Supports various document types including **Technical Documentation**, **Academic Papers**, **Legal Documents**, **Business Reports**, and **General Content**.
- **Usage Context Adaptation**: Tailors responses based on the context of usage such as **Research Analysis**, **Quick Summary**, **Question Answering**, and **Deep Analysis**.
- **Interactive Chat Interface**: Users can ask questions about the content of the uploaded PDFs and receive context-aware responses, enhancing the user experience.

## Installation
To run this application, you need to have **Python** installed on your machine. Follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (if necessary) using a `.env` file.

## Usage
1. Run the application:
   ```bash
   streamlit run pdfchat.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload your PDF documents and select the appropriate document type and usage context.

4. Start interacting with your documents through the chat interface, utilizing the specialized processing capabilities of the application.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
