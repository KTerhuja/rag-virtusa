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
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage




session = st.session_state

# st.title("JLL Assistant")

with st.sidebar:
    st.image('data/JLL_solution_dlow-Page-3.drawio.png')
if 'llm' not in session:
    llm = load_model.load_llm(type='gemini')





if 'generated' not in session:
    session['generated'] = []
## past stores User's questions
if 'past' not in session:
    session['past'] = []
if 'chat_history' not in session:
    session['chat_history'] = []

st.markdown(
    """
<style>
    .st-emotion-cache-janbn0 {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
""",
    unsafe_allow_html=True,
)


with st.spinner('loading vector store'):
    gemini_embedding = embedding.load_embedding('gemini')
    # embedded_vector_store = Chroma(persist_directory='data', embedding_function=gemini_embedding)
    embedded_vector_store = FAISS.load_local("data/faiss", gemini_embedding, allow_dangerous_deserialization=True)


messages = st.container()
user_input = st.chat_input("Query")

# st.write(embedded_vector_store.as_retriever().invoke('ROLES & SECURITY PROPERTIES'))

if user_input:

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, embedded_vector_store.as_retriever(), contextualize_q_prompt
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    # chat_history = []

    output = rag_chain.invoke({"input": user_input, "chat_history": session['chat_history']})
    session['chat_history'].extend([HumanMessage(content=user_input), output["answer"]])
    session['output'] = output
    # st.json(session)

    session.past.append(user_input)
    session.generated.append(output["answer"])
    

if 'generated' in session and user_input:
    with messages:
        for i in range(len(session['generated'])):
            st.chat_message("user", avatar='data/user_icon.png' ).write(session['past'][i])
            st.chat_message("assistant",avatar='data/resize_image (1).png').write(session["generated"][i])

        # st.json(session)

        for idx, doc in enumerate(output['context']):
            filename = os.path.basename(doc.metadata['source'])
            # page_num = doc.metadata['page']
            ref_title = f":blue[Reference {idx}: *{filename}*]"
            with st.popover(ref_title):
                    st.caption(doc.page_content)















 

st.caption("""<style>body {zoom: 80%;}</style>""",unsafe_allow_html=True) 
