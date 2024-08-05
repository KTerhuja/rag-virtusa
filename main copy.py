# Importing necessary modules and functions from different components
from agents import agents
from chunk import splitter
from document_loader import loader
from embedding import embedding
from models import load_model
from retriever import retriever
from vector_store import create_vector_store
from tools import tools
import streamlit as st
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory import ConversationBufferMemory

# Load the document pdf/text/folder of docs
docs = loader.load_docs('data/sample_doc.pdf', type='pdf')

# Split the loaded document into chunks based on character/tokens
split_texts = splitter.split(docs, 'character')

# Create an embedded vector store from the split texts using the openai/gemini embedding type
embedded_vector_store = create_vector_store.create_vector_store(split_texts, embedding_type='gemini')

# Define the input query for retrieval
query = '# input query'

# Retrieve relevant context from the embedded vector store based on the query
retrieve_context = retriever.retrive(embedded_vector_store, query)

# Define the prompt for the language model
prompt = 'prompt'

# Load the language model of type 'gemini'/openai
llm = load_model.load_llm(type='gemini')

# Invoke the language model with the combined prompt and query to get the output
output = llm.invoke(prompt + query)



st.title("JLL Chatbot")
with st.sidebar:
    st.file_uploader(label='upload relevant docs')


# if 'something' not in st.session_state:
#     user_input = ''

# def submit():
#     user_input = st.session_state.widget
#     st.session_state.widget = ''

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = []


if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = ConversationBufferMemory()

if 'chunked_embedding' not in st.session_state:
    docs = loader.load_docs('data/sample_doc.pdf', type='pdf')

    # Split the loaded document into chunks based on character/tokens
    split_texts = splitter.split(docs, 'character')

    # Create an embedded vector store from the split texts using the openai/gemini embedding type
    embedded_vector_store = create_vector_store.create_vector_store(split_texts, embedding_type='gemini')

    st.session_state['vector_store'] = embedded_vector_store



messages = st.container()
user_input = st.chat_input("Query")
relevent_docs = st.expander("Relevent Docs", expanded=False)
if user_input:
    output = '#insert llm output here'
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
if 'generated' in st.session_state:
    with messages:
        for i in range(len(st.session_state['generated'])):
            st.chat_message("user").write(st.session_state['past'][i])
            st.chat_message("assistant").write(st.session_state["generated"][i])


messages = st.container()

user_input = st.chat_input('Query')

relevent_docs = st.expander("Relevent Docs", expanded= False)

if user_input:

    output = 