import streamlit as st
from retrieval import load_and_index_data, retrieve_passages
from generation import generate_answer

# Initialize FAISS index and models (run once)
@st.cache_resource
def initialize():
    return load_and_index_data()

index, passages, tokenizer, model = initialize()

# Streamlit interface
st.title("RAG Question-Answering System")
st.write("Ask a question, and I'll fetch and generate an answer using RAG!")
query = st.text_input("Your Question:", placeholder="e.g., What is machine learning?")

if query:
    with st.spinner("Retrieving and generating answer..."):
        # Retrieve passages
        retrieved_passages = retrieve_passages(query, index, passages, tokenizer, model, k=5)
        
        # Generate answer
        answer = generate_answer(query, retrieved_passages)
        
        # Display results
        st.write("**Answer**:")
        st.write(answer)
        st.write("**Retrieved Passages**:")
        for i, passage in enumerate(retrieved_passages, 1):
            st.write(f"{i}. {passage[:200]}...")  # Truncate for display