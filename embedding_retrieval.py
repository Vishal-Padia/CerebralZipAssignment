import torch
import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

class HybridSearchRetriever:
    def __init__(self, 
                 embedding_model: str = 'microsoft/phi-1_5',
                 embedding_dim: int = 256):
        """
        Initialize hybrid search retriever with support for open-source embedding models.
        
        Args:
            embedding_model (str): Name of the embedding model (e.g., 'microsoft/phi-2').
            embedding_dim (int): Dimensionality of embeddings.
        """
        self.embedding_model = AutoModel.from_pretrained(embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embedding_dim = embedding_dim
        
        # FAISS index for semantic search
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Document storage
        self.documents = []
        self.embeddings = []
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string using the selected embedding model.
        
        Args:
            text (str): Input text to embed.
        
        Returns:
            np.ndarray: Embedding vector.
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Use the [CLS] token embedding or average all token embeddings
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents for semantic search.
        
        Args:
            documents (List[Dict]): List of document chunks.
        """
        # Reset existing index
        self.documents = []
        self.embeddings = []
        
        # Reinitialize FAISS index
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Embed documents
        for doc in documents:
            text = doc['text']
            embedding = self.embed_text(text)
            
            self.documents.append(doc)
            self.embeddings.append(embedding)
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(self.embeddings).astype('float32')
        
        # Add to FAISS index
        self.faiss_index.add(embeddings_array)
        
        # Debug print
        print(f"Indexed {len(self.documents)} documents.")
        print(f"FAISS index contains {self.faiss_index.ntotal} embeddings.")
    
    def semantic_search(self, query: str, k: int = 5):
        """
        Perform semantic search using FAISS.
        
        Args:
            query (str): Search query.
            k (int): Number of results to return.
        
        Returns:
            List of retrieved documents.
        """
        # Embed query
        query_embedding = self.embed_text(query).astype('float32')
        
        # Debug print
        print(f"Query embedding shape: {query_embedding.shape}")
        
        # Semantic search via FAISS
        D, I = self.faiss_index.search(query_embedding.reshape(1, -1), k)
        
        # Debug print
        print(f"FAISS search returned distances: {D}")
        print(f"FAISS search returned indices: {I}")
        
        # Filter out invalid indices (out of range)
        valid_indices = [idx for idx in I[0] if 0 <= idx < len(self.documents)]
        semantic_results = [self.documents[idx] for idx in valid_indices]
        
        # Debug print
        print(f"Semantic search returned {len(semantic_results)} valid results.")
        
        return semantic_results