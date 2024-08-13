# Importing necessary modules and functions from different components
from agents import agents
from splitter import chunk
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
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from prompt_templates import prompts
import os
from streamlit_lottie import st_lottie, st_lottie_spinner
import requests

def save_file(file):
    folder = 'tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = f'./{folder}/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file.getvalue())
    return file_path



st.title("JLL Assistant")
with st.sidebar:
    uploaded_files = st.file_uploader(label='Upload Files')


def render_animation():
    animation_response = requests.get('https://lottie.host/abc043d0-755d-4913-a3b4-bca255815c77/UQfMOeoNgl.json')
    animation_json = dict()
    
    if animation_response.status_code == 200:
        animation_json = animation_response.json()
    else:
        print("Error in the URL")     
                           
    return st_lottie(animation_json, height=75, width=75)

render_animation()

with st_lottie_spinner()

# if 'llm' not in st.session_state:
#     llm = load_model.load_llm(type='gemini')


# if 'chunked_embedding' not in st.session_state:
#     docs = []
#     if uploaded_files:
#         # st.write('flag')
#         file_path = save_file(uploaded_files)
#         # for file in uploaded_files:
#         #     file_path = save_file(file)
#         #     file_loader = loader.load_docs(file_path, 'pdf')
#         #     docs.extend(file_loader.load())
#         # Split the loaded document into chunks based on character/tokens
#         docs = loader.load_docs(file_path, 'pdf')
#         # st.session_state['docs'] = docs
#         split_texts = chunk.split(docs, 'character')

        
#         # st.json(st.session_state)
#      # Create an embedded vector store from the split texts using the openai/gemini embedding type
#         embedded_vector_store = create_vector_store.create_vector_store(split_texts, embedding_type='gemini')

#         st.session_state['vector_store'] = embedded_vector_store


# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []
# ## past stores User's questions
# if 'past' not in st.session_state:
#     st.session_state['past'] = []


# st.markdown(
#     """
# <style>
#     .st-emotion-cache-janbn0 {
#         flex-direction: row-reverse;
#         text-align: right;
#     }
# </style>
# """,
#     unsafe_allow_html=True,
# )
# if not uploaded_files:
#     st.markdown('''##### :gray[Upload files to start chatting]''')

# messages = st.container()
# if uploaded_files:
#     # messages = st.container()
#     user_input = st.chat_input("Query")

#     if user_input:

#         retrieve_context = retriever.retrive(embedded_vector_store, user_input, retrieval_type='mmr')
#         st.session_state['retrived_contex'] = retrieve_context

#         original_chain = ConversationChain(
#             llm=llm,
#             verbose=True,
#             memory=ConversationBufferMemory(memory_key="history")
#         )

#         prompt = prompts._generate_prompt(user_input, retrieve_context)

#         output = original_chain.run(prompt)

#         st.session_state.past.append(user_input)
#         st.session_state.generated.append(output)
        

#     if 'generated' in st.session_state and user_input:
#         with messages:
#             for i in range(len(st.session_state['generated'])):
#                 st.chat_message("user", avatar='data/user_icon.png' ).write(st.session_state['past'][i])
#                 st.chat_message("assistant",avatar='data/resize_image (1).png').write(st.session_state["generated"][i])

#             # st.json(st.session_state)

#             for idx, doc in enumerate(st.session_state['retrived_contex']):
#                 filename = os.path.basename(doc.metadata['file_path'])
#                 page_num = doc.metadata['page']
#                 ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
#                 with st.popover(ref_title):
#                         st.caption(doc.page_content)


st.caption("""<style>body {zoom: 80%;}</style>""",unsafe_allow_html=True) 




# to show references

 



