import streamlit as st
import time
from document_loader import DocumentProcessor
from embedding_retrieval import HybridSearchRetriever
from llm_response import CompactLLMGenerator

class RAGChatApplication:
    def __init__(self):
        """
        Initialize RAG Chat Application with document processing and retrieval
        """
        # Initialize key components
        self.document_processor = DocumentProcessor()
        self.retriever = HybridSearchRetriever()
        self.generator = CompactLLMGenerator()
        
        # Load and index documents only once
        if not st.session_state.get('documents_loaded', False):
            self.load_documents()
            st.session_state['documents_loaded'] = True
    
    def load_documents(self, data_path: str = 'data/en'):
        """
        Load and index documents from specified path
        
        Args:
            data_path (str): Directory containing PDF documents
        """
        with st.spinner('Loading and processing documents...'):
            start_time = time.time()
            
            # Load documents
            documents = self.document_processor.load_pdf_documents(data_path)
            
            # Index documents for retrieval
            self.retriever.index_documents(documents)
            
            # Display processing stats
            end_time = time.time()
            st.success(f"Processed {len(documents)} document chunks in {end_time - start_time:.2f} seconds")
    
    def run_ui(self):
        """
        Run Streamlit User Interface for RAG Chat Application
        """
        # Title and description
        st.title("🤖 RAG Document Chat")
        st.markdown("""
        ### Advanced Retrieval-Augmented Generation (RAG) Chat System
        - Small 2B Model Powered
        - Semantic Search Retrieval
        - Contextual Response Generation
        """)
        
        # Sidebar for configuration and metrics
        st.sidebar.title("RAG System Metrics")
        
        # Chat history state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat input
        user_query = st.chat_input("Ask a question about your documents...")
        
        if user_query:
            # Add user query to chat history
            st.session_state.chat_history.append({
                'role': 'user', 
                'content': user_query
            })
            
            # Retrieval and generation
            with st.spinner('Retrieving and generating response...'):
                start_time = time.time()
                
                # Retrieve contexts
                retrieved_contexts = self.retriever.semantic_search(user_query)
                
                # Generate response
                response = self.generator.generate_response(
                    user_query, 
                    retrieved_contexts, 
                    st.session_state.chat_history
                )
                
                # Evaluate response
                evaluation = self.generator.evaluate_response(
                    user_query, 
                    response, 
                    retrieved_contexts
                )
                
                end_time = time.time()
                
                # Add AI response to chat history
                st.session_state.chat_history.append({
                    'role': 'assistant', 
                    'content': response
                })
                
                # Display metrics in sidebar
                st.sidebar.metric("Retrieval Time", f"{end_time - start_time:.2f} sec")
                st.sidebar.metric("Query Relevance", f"{evaluation['query_relevance']:.2%}")
                st.sidebar.metric("Retrieval Match", f"{evaluation['retrieval_match']:.2%}")
                st.sidebar.metric("Response Fluency", f"{evaluation['fluency']:.2%}")
                
                # Contexts expander
                with st.expander("Retrieved Document Contexts"):
                    for ctx in retrieved_contexts:
                        st.markdown(f"**Source:** {ctx['metadata']['source']}")
                        st.markdown(f"**Page:** {ctx['metadata']['page']}")
                        st.text(ctx['text'][:500] + "...")
                        st.divider()
        
        # Render chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message['role']):
                st.write(message['content'])

def main():
    """
    Main application entry point
    """
    # Set page configuration
    st.set_page_config(
        page_title="RAG Document Chat",
        page_icon="📚",
        layout="wide"
    )
    rag_app = RAGChatApplication()
    rag_app.run_ui()

if __name__ == '__main__':
    main()