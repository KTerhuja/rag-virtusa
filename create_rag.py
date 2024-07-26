# Importing necessary modules and functions from different components
from agents import agents
from chunk import splitter
from document_loader import loader
from embedding import embedding
from models import load_model
from retriever import retriever
from vector_store import create_vector_store
from tools import tools

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
