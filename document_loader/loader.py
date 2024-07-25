from typing import Union
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, DirectoryLoader

def load_docs(path: str, doc_type: str) -> Union[str, list]:
    """
    Load documents from specified path based on document type.

    Args:
        path (str): Path to the document(s) or folder containing documents.
        doc_type (str): Type of document(s) to load. Supported types are 'text', 'pdf', or 'folder'.

    Returns:
        Union[str, list]: Loaded document(s) as a string for single documents ('text', 'pdf'),
                          or a list of strings for multiple text files ('folder').
    
    Raises:
        ValueError: If an unsupported document type is provided.
    """
    if doc_type == 'text':
        loader = TextLoader(path)
    elif doc_type == 'pdf':
        loader = PyMuPDFLoader(path)
    elif doc_type == 'folder':
        loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")

    loaded = loader.load()
    
    return loaded
