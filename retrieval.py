import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

def load_and_index_data(dataset_name="squad_v2", max_samples=100):
    try:
        print("Loading dataset...")
        dataset = load_dataset(dataset_name, split="train")[:max_samples]
        passages = dataset["context"]

        # Add curated machine learning passages
        ml_passages = [
            "Machine learning is a field of artificial intelligence that uses statistical techniques to give computers the ability to learn from data, without being explicitly programmed.",
            "In machine learning, algorithms are trained on data to make predictions or decisions, such as classification, regression, or clustering.",
            "Machine learning includes supervised learning, where models are trained on labeled data, and unsupervised learning, where models find patterns in unlabeled data."
        ]
        passages = list(set(passages + ml_passages))[:max_samples]  # Combine and deduplicate
        print(f"Loaded {len(passages)} passages")

        print("Initializing embedding model...")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        print("Generating embeddings...")
        embeddings = []
        for passage in passages:
            inputs = tokenizer(passage, return_tensors="pt", truncation=True, padding=True, max_length=512)
            embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
            embeddings.append(embedding)
        embeddings = np.vstack(embeddings).astype(np.float32)  # Ensure float32 for FAISS
        
        print("Creating FAISS HNSW index...")
        dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32)  # HNSW with 32 neighbors
        index.hnsw.efConstruction = 200  # Improve index quality
        index.hnsw.efSearch = 40  # Improve search quality
        index.add(embeddings)
        print("Index created")
        return index, passages, tokenizer, model
    except Exception as e:
        print(f"Error in load_and_index_data: {str(e)}")
        raise

def retrieve_passages(query, index, passages, tokenizer, model, k=5):
    try:
        print(f"Processing query: {query}")
        query_input = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
        query_embedding = model(**query_input).last_hidden_state.mean(dim=1).detach().numpy().astype(np.float32)
        distances, indices = index.search(query_embedding, k)
        retrieved = [passages[i] for i in indices[0]]
        print("Retrieved passages:")
        for i, passage in enumerate(retrieved, 1):
            print(f"{i}. {passage[:100]}...")
        return retrieved
    except Exception as e:
        print(f"Error in retrieve_passages: {str(e)}")
        raise