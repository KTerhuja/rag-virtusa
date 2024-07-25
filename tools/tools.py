from langchain_community.tools import DuckDuckGoSearchRun

def invoke_search(query):
    """
    Perform a search using DuckDuckGoSearchRun tool.

    Args:
        query (str): The search query.

    Returns:
        dict: Search results returned by DuckDuckGoSearchRun.
    """
    search = DuckDuckGoSearchRun()
    result = search.run(query)
    return result
