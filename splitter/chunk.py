from langchain_text_splitters import CharacterTextSplitter

def split(docs, split_type):
    """
    Split documents based on the specified splitting method.

    Args:
        docs (str or list): Input documents to split.
        split_type (str): Type of splitting method. 'character' for character-wise split,
                          'token' for token-wise split.

    Returns:
        list: List of split chunks based on the specified split_type.
    """
    if split_type == 'character':
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        chunked_text = text_splitter.split_documents(docs)
        return chunked_text
    
    elif split_type == 'token':
        text_splitter = CharacterTextSplitter.from_token_encoder(chunk_size=100, chunk_overlap=0)
        chunked_tokens = text_splitter.split_documents(docs)
        return chunked_tokens
    
    else:
        raise ValueError(f"Unsupported split type: {split_type}")
