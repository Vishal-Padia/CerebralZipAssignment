# Multilingual PDF RAG System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system designed to process multilingual PDFs, extract information, and provide summaries and answers to questions based on the content. The system is capable of handling both scanned and digitally created PDFs in multiple languages, including Hindi, English, Bengali, and Chinese.

## Features

1. **Text Extraction**:
   - **OCR for Scanned Documents**: Utilizes OCR (Optical Character Recognition) to extract text from scanned PDFs. (**TO BE IMPLEMENTED**)
   - **Standard Extraction for Digital PDFs**: Uses standard PDF parsing techniques for digitally created PDFs.

2. **Advanced RAG System**:
   - **Chat Memory Functionality**: Maintains a conversation history to provide context-aware responses.
   - **Query Decomposition**: Decomposes complex queries into smaller sub-queries for more precise information retrieval.
   - **Optimized Chunking Algorithms**: Splits documents into manageable chunks while maintaining context.
   - **Semantic Search**: Performs semantic search using FAISS for efficient and accurate retrieval.
   - **Integration with Vector Databases**: Supports integration with high-performance vector databases for large-scale data handling.
   - **Selection of LLM and Embedding Models**: Uses compact models like `microsoft/phi-1_5` for efficient and effective processing.
   - **Metadata Filtering**: Filters search results based on metadata such as source and page number.

3. **Evaluation Metrics**:
   - **Query Relevance**: Measures how well the search results match the user's query.
   - **Retrieval Test**: Evaluates whether the retrieved chunks match the context of the query.
   - **Latency**: Measures the time taken to return search results.
   - **Fluency**: Assesses the clarity and coherence of the information presented.
   - **Size of Models**: Demonstrates the use of small embedding models and small LLMs for efficient processing.

## System Architecture

### Components

1. **Document Loader**:
   - **PDF Extraction**: Extracts text from both scanned and digital PDFs.
   - **Chunking**: Splits extracted text into smaller chunks for processing.

2. **Embedding Retrieval**:
   - **Semantic Search**: Uses FAISS for semantic search on document chunks.
   - **Metadata Filtering**: Filters search results based on metadata.

3. **LLM Response Generation**:
   - **Query Decomposition**: Decomposes complex queries into sub-queries.
   - **Response Generation**: Generates responses using a compact LLM.
   - **Evaluation**: Evaluates the quality of generated responses.

4. **User Interface**:
   - **Streamlit App**: Provides a user-friendly interface for interacting with the RAG system.

## Installation and Setup

1. Clone the Repository

2. Install requirements

```bash
pip install -r requirements.txt
```

3. Run the main file
```bash
streamlit run main.py
```

## Future Improvements
1. Reranking Algorithms: Implement reranking algorithms to improve the relevance of search results.

2. Multilingual Support: Enhance the system to support more languages and improve OCR accuracy for non-Latin scripts.

3. Scalability: Optimize the system for handling larger datasets (up to 1TB) by integrating with distributed vector databases.

4. User Interface Enhancements: Improve the Streamlit interface for better user experience and additional features like query history and result filtering.


