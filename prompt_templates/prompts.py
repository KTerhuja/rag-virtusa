
def _generate_prompt(query, context):

    prompt = f'''You are a helpful bot that will answer the user's query
                <query>
                {query}
                </query>

                based on the given context 
                <context>
                {context}
                </context>

                remember to answer the question solely based on the given context and you should not make up 
                answers. Give your response below

                Response:
    
    
                '''

    return prompt