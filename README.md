# RAG Question-Answering System

A Retrieval-Augmented Generation (RAG) system that answers questions by retrieving relevant passages from the SQuAD 2.0 dataset and generating responses using T5. Deployed as a web app using Streamlit.

## Features
- **Retrieval**: Uses FAISS and DistilBERT for efficient passage retrieval.
- **Generation**: Employs T5-small for lightweight, accurate answers.
- **Deployment**: Hosted on Streamlit Cloud (link to be added post-deployment).
- Built with free resources, showcasing Gen AI and LLM skills.

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py