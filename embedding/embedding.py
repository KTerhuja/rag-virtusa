from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def load_embedding(embedding_type):
    """
    Load embeddings based on the specified embedding type.

    Args:
        texts (list): List of texts to embed.
        embedding_type (str): Type of embedding to load. 'open_ai' for OpenAI embeddings,
                              'gemini' for Google Generative AI embeddings.

    Returns:
        object: Embedding model object based on the specified embedding_type.

    Raises:
        ValueError: If an unsupported embedding type is provided.
        KeyError: If the required API key environment variable is not found.
    """
    if embedding_type == 'open_ai':
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise KeyError("Missing environment variable: OPENAI_API_KEY")
        embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
        return embeddings_model
    
    elif embedding_type == 'gemini':
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            raise KeyError("Missing environment variable: GOOGLE_API_KEY")
        gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        return gemini_embeddings
    
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
