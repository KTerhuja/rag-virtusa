def retrive(db, query, retrieval_type):
    """
    Retrieve documents from the database based on the specified retrieval type.

    Args:
        db (Chroma): Chroma object representing the vector store.
        query (str): Query string to retrieve documents relevant to.
        retrieval_type (str): Type of retrieval method. 
            'mmr' for maximum marginal relevance retrieval.
            'similarity_search' for similarity score threshold retrieval.
            'topk' for retrieving top K documents.

    Returns:
        list: List of documents retrieved based on the specified retrieval_type.
    """
    if retrieval_type == 'mmr':
        retriever = db.as_retriever(search_type='mmr')
        docs = retriever.invoke(query)
        return docs
    
    elif retrieval_type == 'similarity_search':
        retriever = db.as_retriever(search_type='similarity_score_threshold', search_kwargs={"score_threshold": 0.5})
        docs = retriever.invoke(query)
        return docs
    
    elif retrieval_type == 'topk':
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        return docs
    
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
