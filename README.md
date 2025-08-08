# RAG Question-Answering System

A Retrieval-Augmented Generation (RAG) system that answers questions by retrieving relevant passages and generating responses using state-of-the-art NLP models. Built with free resources, this project combines semantic search with text generation, ideal for educational and portfolio purposes.

## Features
- **Semantic Retrieval**: Uses `sentence-transformers/all-MiniLM-L6-v2` for accurate passage retrieval via FAISS with HNSW indexing.
- **Answer Generation**: Employs `t5-small` for lightweight, precise answer generation from retrieved passages.
- **Curated Dataset**: Enhances SQuAD 2.0 with handcrafted machine learning passages for better accuracy.
- **Web Interface**: Deployed via Streamlit, with support for Google Colab using `ngrok`.
- **Portfolio-Ready**: Demonstrates end-to-end Gen AI/LLM skills, building on my experience with [DocGenAI](https://github.com/superuser303/DocGenAI).

## Demo
Ask questions like "What is machine learning?" to get accurate, concise answers backed by relevant passages. (Live demo link to be added post-deployment.)

## Setup and Installation
### Prerequisites
- Python 3.8–3.12
- Google Colab (for free-tier usage) or local environment
- Free ngrok account (for Colab Streamlit hosting)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/superuser303/RAG-Question-Answering.git
   cd RAG-Question-Answering
