from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()


def load_llm(type):

    if type == 'openai':
        openai_api_key = os.getenv('OPENAI_API_KEY')

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=openai_api_key,  # if you prefer to pass api key in directly instaed of using env vars
        )

        return llm
    
    elif type == 'gemini':

        google_api_key = os.getenv('GOOGLE_API_KEY')

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key = google_api_key
        )

        return llm