import os
import re
import tiktoken
import warnings
warnings.filterwarnings("ignore")

from fastapi import HTTPException
from dotenv import load_dotenv
from typing import List

from handlers.sql_agent import sql_agent
from handlers.openai_tools_agent import openai_tools_agent
from handlers.sql_query_chain import sql_query_chain
from handlers.react_agent import react_agent
from utils.alerts import alert_exception, alert_info

from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.structured import  StructuredOutputParser, ResponseSchema
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

class BaseHandler():
    def __init__(
            self,
            chat_model: str = 'gpt-4o-mini',
            temperature: float = 0.2,
            **kwargs
        ):

        self.llm_map = {
            'gpt-4o-mini': lambda _: ChatOpenAI(model='gpt-4o-mini', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'gpt-4': lambda _: ChatOpenAI(model='gpt-4', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'gpt-4-32k': lambda _: ChatOpenAI(model='gpt-4-32k', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'gpt-4-1106-preview': lambda _: ChatOpenAI(model='gpt-4', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'gpt-3.5-turbo-16k': lambda _: ChatOpenAI(model='gpt-3.5-turbo-16k', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'gpt-3.5-turbo': lambda _: ChatOpenAI(model='gpt-3.5-turbo', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'claude-3-sonnet-20240229': lambda _: ChatAnthropic(model_name='claude-3-sonnet-20240229', temperature=temperature, anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')),
            'claude-3-opus-20240229': lambda _: ChatAnthropic(model_name='claude-3-opus-20240229', temperature=temperature, anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')),
        }

        self.chat_model = chat_model
               
    def chat(self, connection_string: str, user_question: str, chat_history: list[str] = []):
        """
        connection_string: str
        user_question: str
        chat_history: list of previous chat messages
        kwargs:
            namespace: str
            search_kwargs: dict
        """
        try: 
            # ****************************************************
            # Connecting to the DB
            # ****************************************************
            connection_string=os.getenv('DB_CONNECTION_STRING')
            db = SQLDatabase.from_uri(connection_string)

            # validate the connection to the db
            # print(db.dialect)
            # print(db.get_usable_table_names())
            # response = db.run("SELECT TOP 10 * FROM T130Regioni")

            # ****************************************************
            # Get LLM model
            # ****************************************************
            # llm=self.llm_map[self.chat_model] 
            
            # model='gpt-3.5-turbo'
            model='gpt-4o-mini'

            llm=ChatOpenAI(
                 model=model, 
                 temperature=0.2, 
                 openai_api_key=os.getenv('OPENAI_API_KEY'), 
                 verbose=True)

            # ****************************************************
            # Answer the questions
            # ****************************************************

            # OK    
            # response=sql_query_chain(user_question, db, llm)

            # OK    
            response=react_agent(user_question, db, llm)

            # Fail
            # response=sql_agent(user_question, db, llm)

            # Fail
            # response=openai_tools_agent(user_question, db, llm)

            return response
        
        except Exception as e:
            alert_exception(e, "Error chatting")
            raise HTTPException(status_code=500, detail=f"Error chatting: {str(e)}")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens