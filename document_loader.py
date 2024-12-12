import os
import torch
import nltk
from typing import List, Dict
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)

class DocumentProcessor:
    def __init__(self, 
                 chunk_size: int = 256, 
                 chunk_overlap: int = 64,
                 metadata_fields: List[str] = ['source', 'page']):
        """
        Initialize document processor with configurable chunking parameters
        
        Args:
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks to maintain context
            metadata_fields (List[str]): Metadata to extract from documents
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata_fields = metadata_fields
        
        # Advanced text splitter with multi-level splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
    
    def load_pdf_documents(self, directory: str) -> List[Dict]:
        """
        Load PDF documents from a directory with advanced parsing
        
        Args:
            directory (str): Path to directory containing PDFs
        
        Returns:
            List of document chunks with metadata
        """
        documents = []
        
        # Scan all PDF files in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                filepath = os.path.join(directory, filename)
                
                # Use PyPDF2 for robust PDF reading
                try:
                    pdf_reader = PdfReader(filepath)
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        # Extract text with error handling
                        try:
                            page_text = page.extract_text()
                        except Exception as e:
                            print(f"Error extracting text from {filename}, page {page_num}: {e}")
                            continue
                        
                        # Split text into chunks
                        chunks = self.text_splitter.split_text(page_text)
                        
                        # Add metadata to each chunk
                        for chunk in chunks:
                            documents.append({
                                'text': chunk,
                                'metadata': {
                                    'source': filename,
                                    'page': page_num
                                }
                            })
                
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return documents
    
    def preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing
        
        Args:
            text (str): Input text to preprocess
        
        Returns:
            Cleaned and normalized text
        """
        # Lowercase conversion
        text = text.lower()
        
        # Remove excess whitespace
        text = ' '.join(text.split())
        
        return text
    
    def add_random_noise(self, embeddings: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        Add small random noise to embeddings for robustness
        
        Args:
            embeddings (np.ndarray): Input embeddings
            noise_level (float): Level of noise to add
        
        Returns:
            Embeddings with added noise
        """
        noise = np.random.normal(0, noise_level, embeddings.shape)
        return embeddings + noise
