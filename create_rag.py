from agents import agents
from chunk import splitter
from document_loader import loader
from embedding import embedding
from models import load_model
from retriever import retriever
from vector_store import create_vector_store
from tools import tools


docs = loader.load_docs('data/sample_doc.pdf', type='pdf')

split_texts = splitter.split(docs, 'character')

embedded_vector_store = create_vector_store.create_vector_store(split_texts, embedding_type='gemini')

query = '# input query'

retrieve_context = retriever.retrive(embedded_vector_store, query)

prompt = 'prompt'

llm = load_model.load_llm(type='gemini')




