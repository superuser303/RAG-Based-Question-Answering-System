RAG Question-Answering System

A Retrieval-Augmented Generation (RAG) system that answers questions by retrieving relevant passages and generating responses using state-of-the-art NLP models. Built with free resources, this project combines semantic search with text generation, ideal for educational and portfolio purposes.

Features





Semantic Retrieval: Uses sentence-transformers/all-MiniLM-L6-v2 for accurate passage retrieval via FAISS with HNSW indexing.



Answer Generation: Employs t5-small for lightweight, precise answer generation from retrieved passages.



Curated Dataset: Enhances SQuAD 2.0 with handcrafted machine learning passages for better accuracy.



Web Interface: Deployed via Streamlit, with support for Google Colab using ngrok.



Portfolio-Ready: Demonstrates end-to-end Gen AI/LLM skills, building on my experience with DocGenAI.

Demo

Ask questions like "What is machine learning?" to get accurate, concise answers backed by relevant passages. (Live demo link to be added post-deployment.)

Setup and Installation

Prerequisites





Python 3.8–3.12



Google Colab (for free-tier usage) or local environment



Free ngrok account (for Colab Streamlit hosting)

Steps





Clone the Repository:

git clone https://github.com/superuser303/RAG-Question-Answering.git
cd RAG-Question-Answering



Install Dependencies:

pip install -r requirements.txt

Contents of requirements.txt:

faiss-cpu==1.8.0
numpy==1.26.4
transformers==4.44.2
datasets==3.0.1
streamlit==1.39.0
torch==2.4.1
sentence-transformers==2.7.0
pyngrok==7.1.6



Run the App:





In Colab:

!pip install pyngrok streamlit
from pyngrok import ngrok
!ngrok authtoken YOUR_AUTHTOKEN  # Get from https://dashboard.ngrok.com
public_url = ngrok.connect(8501)
print(f"Streamlit app at: {public_url}")
!streamlit run app.py --server.port 8501

Open the ngrok URL in a browser.



Locally:

streamlit run app.py

Open localhost:8501 in a browser.



Ask Questions: Enter queries like "What is machine learning?" to see answers and retrieved passages.

Project Structure

RAG-Question-Answering/
├── app.py          # Streamlit web interface
├── retrieval.py    # Semantic retrieval with FAISS and MiniLM
├── generation.py   # Answer generation with T5
├── requirements.txt # Dependencies
└── README.md       # Project documentation

How It Works





Retrieval: Uses sentence-transformers/all-MiniLM-L6-v2 to embed passages from SQuAD 2.0 and curated ML contexts, indexed with FAISS HNSW for fast semantic search.



Generation: Feeds the top 5 retrieved passages and query to t5-small to generate concise answers.



Interface: Streamlit provides a user-friendly UI to input questions and view results.

Results

For "What is machine learning?", the system retrieves passages like:



Machine learning is a field of artificial intelligence that uses statistical techniques...

And generates answers like:



Machine learning is a field of AI that enables computers to learn from data without explicit programming.

Future Improvements





Fine-tune t5-small on domain-specific Q&A data.



Expand the dataset with more technical articles.



Deploy to Streamlit Cloud for a public demo.

About

Built by superuser303 to showcase Gen AI and LLM skills. Inspired by my work on DocGenAI, this project explores RAG for real-world NLP applications.
