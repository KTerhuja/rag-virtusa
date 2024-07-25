from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent

def create_react_agent_executor(llm, tools, prompt, query):
    """
    Create and execute a ReAct agent.

    Args:
        llm: The language model for the agent.
        tools: Tools required by the agent.
        prompt: Prompt to initialize the agent.
        query (dict): Query input for the agent.

    Returns:
        dict: Output from the ReAct agent execution.
    """
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_output = agent_executor.invoke({"input": query})
    return agent_output

def create_structured_chat_agent_executor(llm, tools, prompt, query):
    """
    Create and execute a structured chat agent.

    Args:
        llm: The language model for the agent.
        tools: Tools required by the agent.
        prompt: Prompt to initialize the agent.
        query (dict): Query input for the agent.

    Returns:
        dict: Output from the structured chat agent execution.
    """
    agent = create_structured_chat_agent(agent=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_output = agent_executor.invoke({"input": query})
    return agent_output
