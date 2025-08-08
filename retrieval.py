import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

def load_and_index_data(dataset_name="squad_v2", max_samples=50):  # Reduced for Colab
    try:
        print("Loading dataset...")
        dataset = load_dataset(dataset_name, split="train")[:max_samples]
        passages = dataset["context"]
        print(f"Loaded {len(passages)} passages")

        print("Initializing embedding model...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        
        print("Generating embeddings...")
        embeddings = []
        for passage in passages:
            inputs = tokenizer(passage, return_tensors="pt", truncation=True, padding=True)
            embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
            embeddings.append(embedding)
        embeddings = np.vstack(embeddings)
        
        print("Creating FAISS index...")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        print("Index created")
        return index, passages, tokenizer, model
    except Exception as e:
        print(f"Error in load_and_index_data: {str(e)}")
        raise

def retrieve_passages(query, index, passages, tokenizer, model, k=5):
    try:
        query_input = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        query_embedding = model(**query_input).last_hidden_state.mean(dim=1).detach().numpy()
        distances, indices = index.search(query_embedding, k)
        return [passages[i] for i in indices[0]]
    except Exception as e:
        print(f"Error in retrieve_passages: {str(e)}")
        raise