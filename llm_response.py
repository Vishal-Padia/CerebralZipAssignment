import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
from document_loader import DocumentProcessor
from embedding_retrieval import HybridSearchRetriever

class CompactLLMGenerator:
    def __init__(self, 
                 model_name: str = 'microsoft/phi-1_5', 
                 max_length: int = 1024):
        """
        Initialize compact LLM for response generation
        
        Args:
            model_name (str): Compact language model name
            max_length (int): Maximum generation length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32,  # Use CPU
        )
        
        self.max_length = max_length
        
        # Conversation memory
        self.conversation_history = []
        self.max_history_length = 5  # Limit to prevent context bloat
    
    def generate_response(
        self, 
        query: str, 
        retrieved_contexts: List[Dict], 
        chat_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate a response using retrieved contexts and chat history
        
        Args:
            query (str): User's query
            retrieved_contexts (List[Dict]): Relevant document chunks
            chat_history (Optional[List[Dict]]): Previous conversation turns
        
        Returns:
            Generated response
        """
        # Prepare context
        context_str = "\n\n".join([
            f"Source: {ctx['metadata']['source']}, Page: {ctx['metadata']['page']}\n{ctx['text']}" 
            for ctx in retrieved_contexts
        ])
        
        # Prepare chat history
        history_str = ""
        if chat_history:
            history_str = "\n".join([
                f"Human: {turn.get('query', '')}\nAI: {turn.get('response', '')}" 
                for turn in chat_history[-self.max_history_length:]
            ])
        
        # Construct prompt with context and history
        prompt = f"""You are a helpful AI assistant. Use only the provided context to answer the query.
        
        Chat History:
        {history_str}
        
        Retrieved Context:
        {context_str}
        
        User Query: {query}
        
        Detailed Response:"""
        
        # Tokenize and generate
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Update conversation history
        self.conversation_history.append({
            'query': query,
            'response': response,
            'retrieved_contexts': retrieved_contexts
        })
        
        # Trim conversation history if needed
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        return response