from splitter import chunk
from document_loader import loader
from embedding import embedding
# import pysqlite3
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

def create_vector_store(texts, embedding_type):
    """
    Create a vector store from documents based on the specified embedding type.

    Args:
        texts (list): List of texts to create vectors for.
        embedding_type (str): Type of embedding to use. 'open_ai' for OpenAI embeddings,
                              'gemini' for Google Generative AI embeddings.

    Returns:
        Chroma: Vector store created using the specified embedding type.
    
    Raises:
        ValueError: If an unsupported embedding type is provided.
    """
    if embedding_type == 'open_ai':
        openai_embedding = embedding.load_embedding('open_ai')
        # db = Chroma.from_documents(texts, openai_embedding, persist_directory= "vector_store")
        return db
    
    elif embedding_type == 'gemini':
        gemini_embedding = embedding.load_embedding('gemini')
        # db = Chroma.from_documents(texts, gemini_embedding, persist_directory= "./data")

        db = FAISS.from_documents(texts, gemini_embedding)
        db.save_local("data/faiss")
    
    
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    

if __name__ == "__main__":

    # Load the document pdf/text/folder of docs
    docs = loader.load_docs('data/delio', type='folder')

    # Split the loaded document into chunks based on character/tokens
    split_texts = chunk.split(docs, 'character')

    # Create an embedded vector store from the split texts using the openai/gemini embedding type
    embedded_vector_store = create_vector_store(split_texts, embedding_type='gemini')

    create_vector_store(split_texts, embedding_type='gemini')
