import os
import streamlit as st
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# ==========================================
# ERROR HANDLING WRAPPER
# ==========================================
try:
    # Load environment variables
    load_dotenv()

    # Verify Gemini API key is present before loading UI
    if not os.getenv('GEMINI_API_KEY'):
        st.error('⚠️ Error: GEMINI_API_KEY not found. Make sure your .env file has GEMINI_API_KEY set.')
        st.stop()

    # ==========================================
    # PAGE CONFIGURATION & INITIALIZATION
    # ==========================================
    st.set_page_config(
        page_title='AI Assistant (Gemini)',
        page_icon='🤖',
        layout='wide'
    )

    st.title('Company Knowledge Assistant')
    st.markdown('Ask me anything about company policies! (Powered by Google Gemini)')

    # Initialize ChromaDB with Gemini Embeddings (Cached)
    @st.cache_resource
    def init_chromadb():
        if not os.path.exists('./chroma_db'):
            raise FileNotFoundError()
            
        client = chromadb.PersistentClient(path='./chroma_db')
        
        # Configure Gemini-native embedding mapping helper
        gemini_ef = embedding_functions.GoogleGeminiEmbeddingFunction(
            api_key=os.getenv('GEMINI_API_KEY'),
            model_name="text-embedding-004" # Standard Google embedding model
        )
        
        collection = client.get_collection(
            name='company_docs',
            embedding_function=gemini_ef
        )
        return collection

    # Initialize Gemini LLM via LangChain (Cached)
    @st.cache_resource
    def init_llm():
        return ChatGoogleGenerativeAI(
            model='gemini-1.5-flash',
            temperature=0
        )

    # Initialize assets
    collection = init_chromadb()
    llm = init_llm()

    # ==========================================
    # RAG FUNCTION (Task 2.3 Adaptation)
    # ==========================================
    def get_rag_response(query, n_results=3):
        try:
            # Query vector database using query text strings
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if not results or not results['documents'] or not results['documents'][0]:
                return 'No relevant information found in documents.'
            
            # Combine documents into a context block
            context = '\n\n---\n\n'.join(results['documents'][0])
            
            # Construct strict prompt framing instructions
            prompt = (
                f"You are a helpful HR assistant. Answer using ONLY the context provided below. "
                f"If the information is not explicitly contained within the context, say so clearly. "
                f"Be concise, friendly, and accurate.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )
            
            # Invoke Gemini chat frame structure
            response = llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f'Error generating response: {str(e)}. Please check your database matching configuration.'

    # ==========================================
    # CHAT HISTORY SESSION STATE
    # ==========================================
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # ==========================================
    # SIDEBAR UI COMPONENT
    # ==========================================
    with st.sidebar:
        st.header('About')
        st.markdown("""
        This AI assistant can answer questions about:
        * Remote work guidelines
        * Parental leave
        * Vacation policies
        * Benefits information
        
        **Engineered with:**
        * Google Gemini 1.5 Flash
        * ChromaDB Vector Search
        * LangChain Integration
        """)
        
        st.divider()
        st.metric('Documents Indexed', collection.count())
        st.metric('Messages in Chat', len(st.session_state.messages))
        st.divider()
        
        if st.button('Clear Chat History'):
            st.session_state.messages = []
            st.sidebar.success("History wiped!")
            st.rerun()

    # ==========================================
    # MAIN CHAT APPLICATION STREAM
    # ==========================================
    if len(st.session_state.messages) == 0:
        welcome = (
            "Hi! I'm your company knowledge assistant. "
            "I can help you find information about company guidelines, parental benefits, or "
            "vacation and time-off structures. Ask away!"
        )
        with st.chat_message('assistant'):
            st.write(welcome)

    # Render persistent conversation elements
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])

    # Input capture logic
    if prompt := st.chat_input('Ask a question...'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.write(prompt)
        
        # Loading phase while calling vector DB + Gemini API
        with st.chat_message('assistant'):
            with st.spinner('Searching documents...'):
                response = get_rag_response(prompt)
                st.write(response)
        
        st.session_state.messages.append({'role': 'assistant', 'content': response})

except FileNotFoundError:
    st.error('❌ Error: ChromaDB database directory not found. Please verify your vector database exists.')
    st.stop()
except Exception as e:
    st.error(f'⚠️ Critical Application Error: {str(e)}')
    st.stop()
