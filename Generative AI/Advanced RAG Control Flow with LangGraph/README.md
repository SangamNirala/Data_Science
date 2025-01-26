# Advanced RAG control flow with LangGraph🦜🕸:

Implementation of Reflective RAG, Self-RAG & Adaptive RAG tailored towards developers and production-oriented applications for learning LangGraph🦜🕸️.

This repository contains a refactored version of the original [LangChain's Cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/langchain),

See Original YouTube video:[Advance RAG control flow with Mistral and LangChain](https://www.youtube.com/watch?v=sgnrL7yo1TE)

of [Sophia Young](https://x.com/sophiamyang) from Mistral & [Lance Martin](https://x.com/RLanceMartin) from LangChain

![Logo](https://github.com/emarco177/langgaph-course/blob/main/static/langgraph_adaptive_rag.png)
[![udemy](https://img.shields.io/badge/LangGraph🦜🔗%20Udemy%20Course-ODSC%20Coupon%20%2412.99-brightgreen)](https://www.udemy.com/course/langgraph/?couponCode=ODSC-2024-DB8B797EAD)

## Features

- **Refactored Notebooks**: The original LangChain notebooks have been refactored to enhance readability, maintainability, and usability for developers.
- **Production-Oriented**: The codebase is designed with a focus on production readiness, allowing developers to seamlessly transition from experimentation to deployment.
- **Test Coverage**: Comprehensive test coverage ensures the reliability and stability of the application, enabling developers to validate their implementations effectively.
- **Documentation**: Detailed documentation and branches guide developers through setting up the environment, understanding the codebase, and utilizing LangGraph effectively.

## Project Structure
The project is organized into several key components:

- **ingestion.py**: Responsible for loading and processing documents from specified URLs into a vector store.
- **main.py**: The entry point of the application that initializes the workflow and handles user queries.
- **graph/**: Contains the core logic of the graph-based application, including:
  - **consts.py**: Defines constants for node names to avoid code duplication.
  - **state.py**: Manages the state of the graph during execution.
  - **nodes/**: Contains implementations of various nodes, including:
    - **generate.py**: Handles the generation of responses based on retrieved documents.
    - **grade_documents.py**: Evaluates the relevance of retrieved documents.
    - **retrieve.py**: Retrieves relevant documents based on user queries.
    - **web_search.py**: Performs web searches using the Tavily search engine.
  - **chains/**: Contains the logic for chaining nodes together, including:
    - **answer_grader.py**: Grades the relevance of answers.
    - **generation.py**: Manages the generation of responses.
    - **hallucination_grader.py**: Checks for hallucinations in generated responses.
    - **router.py**: Routes questions to the appropriate nodes based on conditions.

## Workflow
1. **Ingestion**: Load documents from specified URLs and index them into a vector store.
2. **Retrieval**: Retrieve relevant documents based on user queries.
3. **Grading**: Evaluate the relevance of retrieved documents and filter out irrelevant ones.
4. **Web Search**: If necessary, perform a web search to find additional relevant information.
5. **Generation**: Generate a response based on the relevant documents and the original question.

## Dependencies
- LangChain
- Chroma
- OpenAI API
- Tavily API

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file:

- `PYTHONPATH=/{YOUR_PATH_TO_PROJECT}/langgraph-course`
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

## Setup
1. Clone the repository.
   ```bash
   git clone https://github.com/emarco177/langgraph-course.git
   ```
2. Go to the project directory.
   ```bash
   cd langgraph-course
   ```
3. Install dependencies.
   ```bash
   poetry install
   ```
4. Start the flask server.
   ```bash
   poetry run main.py
   ```

## Running Tests

To run tests, run the following command:
```bash
poetry run pytest . -s -v
```

## Acknowledgements

Original LangChain repository: [LangChain Cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/langchain)  
By [Sophia Young](https://x.com/sophiamyang) from Mistral & [Lance Martin](https://x.com/RLanceMartin) from LangChain  
![Logo](https://github.com/emarco177/langgaph-course/blob/main/static/LangChain-logo.png)
