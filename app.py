import streamlit as st
from retrieval import load_and_index_data, retrieve_passages
from generation import generate_answer

@st.cache_resource
def initialize():
    print("Initializing...")
    return load_and_index_data(max_samples=100)  # Match retrieval.py

try:
    index, passages, tokenizer, model = initialize()
    print("Initialization complete")
except Exception as e:
    print(f"Initialization error: {str(e)}")
    st.error(f"Failed to initialize: {str(e)}")
    st.stop()

st.title("RAG Question-Answering System")
st.write("Ask a question, and I'll fetch and generate an answer using RAG!")
query = st.text_input("Your Question:", placeholder="e.g., What is machine learning?")

if query:
    with st.spinner("Retrieving and generating answer..."):
        try:
            print("Retrieving passages...")
            retrieved_passages = retrieve_passages(query, index, passages, tokenizer, model, k=5)
            print(f"Retrieved {len(retrieved_passages)} passages")
            answer = generate_answer(query, retrieved_passages)
            if answer:
                st.write("**Answer**:")
                st.write(answer)
                st.write("**Retrieved Passages**:")
                for i, passage in enumerate(retrieved_passages, 1):
                    st.write(f"{i}. {passage[:200]}...")
            else:
                st.error("Failed to generate answer.")
        except Exception as e:
            print(f"Error in query processing: {str(e)}")
            st.error(f"Error: {str(e)}")