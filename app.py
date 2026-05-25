import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# ==========================================
# ERROR HANDLING WRAPPER (Task 3.3)
# ==========================================
try:
    # Load environment variables
    load_dotenv() # [cite: 44]

    # Quick check for API key before spinning up the app
    if not os.getenv('OPENAI_API_KEY'):
        st.error('⚠️ Error: OPENAI_API_KEY not found in your .env file.') # 
        st.stop()

    # ==========================================
    # PAGE CONFIGURATION & INITIALIZATION
    # ==========================================
    st.set_page_config(
        page_title='AI Assistant', # [cite: 29]
        page_icon='🤖',
        layout='wide' # [cite: 30]
    )

    st.title('Company Knowledge Assistant') # [cite: 30]
    st.markdown('Ask me anything about company policies!') # [cite: 30]

    # Cache ChromaDB client connection so it runs once (Task 2.2)
    @st.cache_resource # [cite: 46]
    def init_chromadb():
        # Check if directory exists before connecting
        if not os.path.exists('./chroma_db'):
            raise FileNotFoundError() # [cite: 141]
            
        client = chromadb.PersistentClient(path='./chroma_db') # [cite: 47]
        openai_ef = embedding_functions.OpenAIEmbeddingFunction( # [cite: 48, 53]
            api_key=os.getenv('OPENAI_API_KEY'), # [cite: 49]
            model_name='text-embedding-ada-002' # [cite: 54]
        )
        collection = client.get_collection( # [cite: 55]
            name='company_docs', # [cite: 51]
            embedding_function=openai_ef # [cite: 56]
        )
        return collection

    # Cache LLM client so it runs once (Task 2.2)
    @st.cache_resource # [cite: 57]
    def init_llm():
        return ChatOpenAI( # [cite: 63]
            model='gpt-3.5-turbo', # [cite: 67]
            temperature=0 # [cite: 62]
        )

    # Initialize the resources
    collection = init_chromadb() # [cite: 69]
    llm = init_llm() # [cite: 70]

    # ==========================================
    # RAG FUNCTION (Task 2.3)
    # ==========================================
    def get_rag_response(query, n_results=3): # [cite: 72]
        try:
            # Query vector database
            results = collection.query( # [cite: 73]
                query_texts=[query], # [cite: 76]
                n_results=n_results # [cite: 77]
            )
            
            # Check if any documents were retrieved
            if not results or not results['documents'] or not results['documents'][0]: # [cite: 79, 81]
                return 'No relevant information found in documents.' # [cite: 82]
            
            # Build context strings
            context = '\n\n---\n\n'.join(results['documents'][0]) # [cite: 83]
            
            # Construct strict prompt
            prompt = (
                f"You are a helpful HR assistant. Answer using ONLY the context below. "
                f"If not in context, say so. Be concise and friendly.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            ) # [cite: 84]
            
            messages = [{'role': 'user', 'content': prompt}] # [cite: 85]
            
            # Invoke LLM
            response = llm.invoke(messages) # [cite: 86]
            return response.content # [cite: 89]
            
        except Exception as e:
            return f'Error: {str(e)}. Please try again.' # [cite: 90]

    # ==========================================
    # CHAT HISTORY SESSION STATE
    # ==========================================
    if 'messages' not in st.session_state: # [cite: 30]
        st.session_state.messages = [] # [cite: 30]

    # ==========================================
    # SIDEBAR UI COMPONENT (Task 3.1)
    # ==========================================
    with st.sidebar: # [cite: 103]
        st.header('About') # [cite: 105]
        st.markdown("""
        This AI assistant can answer questions about:
        * Remote work guidelines
        * Parental leave
        * Vacation policies
        * Benefits information
        
        **Powered by:**
        * OpenAI GPT-3.5
        * ChromaDB vector search
        * Semantic RAG
        """) # [cite: 104, 106, 107, 108, 109, 110, 115, 116, 117, 119]
        
        st.divider() # [cite: 111]
        
        # Display live application statistics
        st.metric('Documents Indexed', collection.count()) # [cite: 112, 120]
        st.metric('Messages in Chat', len(st.session_state.messages)) # [cite: 113, 121]
        
        st.divider() # [cite: 125]
        
        # Clear Chat Button logic
        if st.button('Clear Chat History'): # [cite: 122, 126]
            st.session_state.messages = [] # [cite: 123, 124]
            st.sidebar.success("History wiped!")
            st.rerun() # [cite: 127]

    # ==========================================
    # MAIN INTERFACE CHAT STREAM
    # ==========================================
    # Show welcome prompt on clean slate (Task 3.2)
    if len(st.session_state.messages) == 0: # [cite: 131]
        welcome = (
            "Hi! I'm your company knowledge assistant. "
            "I can help you find information about:\n"
            "- Remote work guidelines\n"
            "- Parental leave benefits\n"
            "- Vacation and time off policies\n"
            "And more! Just ask me a question to get started."
        ) # [cite: 132, 133, 134, 135, 136]
        with st.chat_message('assistant'): # [cite: 137]
            st.write(welcome) # [cite: 138]

    # Display prior conversational history elements
    for message in st.session_state.messages: # [cite: 31]
        with st.chat_message(message['role']): # [cite: 32, 33]
            st.write(message['content']) # [cite: 34]

    # Accept new user queries
    if prompt := st.chat_input('Ask a question...'): # [cite: 34]
        # Append and show user input immediately
        st.session_state.messages.append({'role': 'user', 'content': prompt}) # [cite: 34]
        with st.chat_message('user'): # [cite: 34]
            st.write(prompt) # [cite: 34]
        
        # Process and show assistant response within a loading state
        with st.chat_message('assistant'): # [cite: 92]
            with st.spinner('Searching documents...'): # [cite: 94, 97]
                response = get_rag_response(prompt) # [cite: 98]
                st.write(response) # [cite: 95]
        
        # Keep assistant answer in ongoing conversation record
        st.session_state.messages.append({'role': 'assistant', 'content': response}) # [cite: 99]

except FileNotFoundError: # [cite: 141]
    st.error('❌ Error: ChromaDB not found. Please run Week 11 lab to create the vector database first.') # [cite: 141]
    st.stop() # [cite: 141]
except Exception as e: # [cite: 141]
    st.error(f'⚠️ Critical Application Error: {str(e)}') # [cite: 141]
    st.stop() # [cite: 141]
