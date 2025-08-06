import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

def load_and_index_data(dataset_name="squad_v2", max_samples=1000):
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")[:max_samples]
    passages = [example["context"] for example in dataset]
    
    # Initialize embedding model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    
    # Generate embeddings
    embeddings = []
    for passage in passages:
        inputs = tokenizer(passage, return_tensors="pt", truncation=True, padding=True)
        embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding)
    embeddings = np.vstack(embeddings)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return index, passages, tokenizer, model

def retrieve_passages(query, index, passages, tokenizer, model, k=5):
    # Embed query
    query_input = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    query_embedding = model(**query_input).last_hidden_state.mean(dim=1).detach().numpy()
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, k)
    return [passages[i] for i in indices[0]]