# RAG Question-Answering System

A Retrieval-Augmented Generation (RAG) system that answers questions like "What is machine learning?" by retrieving relevant passages and generating concise responses using NLP models. Built with free resources, this project showcases semantic search and text generation for my GitHub portfolio.

## Features
- **Semantic Retrieval**: Uses `sentence-transformers/all-MiniLM-L6-v2` with FAISS HNSW indexing for accurate passage retrieval.
- **Answer Generation**: Employs `t5-small` for lightweight, precise answers.
- **Curated Dataset**: Enhances SQuAD 2.0 with handcrafted machine learning passages.
- **Web Interface**: Runs via Streamlit, with Google Colab support using `ngrok`.
- **Portfolio-Ready**: Demonstrates Gen AI/LLM skills, building on my [DocGenAI](https://github.com/superuser303/DocGenAI) experience.

## Demo
Enter queries like "What is machine learning?" to get answers like:  
> Machine learning is a field of AI that enables computers to learn from data without explicit programming.  
(Live demo link to be added post-deployment to Streamlit Cloud.)

## How to Run the Project
Follow these steps to set up and run the project after downloading the repository.

### Prerequisites
- Python 3.8–3.12
- Google Colab (for free-tier usage) or local environment
- Free ngrok account (for Colab, get authtoken from [ngrok.com](https://dashboard.ngrok.com))

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/superuser303/RAG-Based-Question-Answering-System.git
   cd RAG-Based-Question-Answering-System
