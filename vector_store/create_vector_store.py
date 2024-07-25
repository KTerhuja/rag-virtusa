from langchain_chroma import Chroma
from embedding import embedding

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
        openai_embedding = embedding.load_embedding(texts, 'open_ai')
        db = Chroma.from_documents(texts, openai_embedding)
        return db
    
    elif embedding_type == 'gemini':
        gemini_embedding = embedding.load_embedding(texts, 'gemini')
        db = Chroma.from_documents(texts, gemini_embedding)
        return db
    
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
