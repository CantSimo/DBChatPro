import os
import tiktoken
import warnings
warnings.filterwarnings("ignore")

from operator import itemgetter
from utils.alerts import alert_exception, alert_info
from typing import List
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder,
    PromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import (
    LLMChain, 
    ConversationalRetrievalChain, 
    create_history_aware_retriever, 
    create_retrieval_chain, 
    create_sql_query_chain)
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from fastapi import UploadFile
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()

class BaseHandler():
    def __init__(
            self,
            chat_model: str = 'gpt-3.5-turbo',
            temperature: float = 0.2,
            **kwargs
        ):

        self.charts_folder = os.getenv('CHARTS_FOLDER')
        self.llm_map = {
            'gpt-4': lambda _: ChatOpenAI(model='gpt-4', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'gpt-4-32k': lambda _: ChatOpenAI(model='gpt-4-32k', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'gpt-4-1106-preview': lambda _: ChatOpenAI(model='gpt-4', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'gpt-3.5-turbo-16k': lambda _: ChatOpenAI(model='gpt-3.5-turbo-16k', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'gpt-3.5-turbo': lambda _: ChatOpenAI(model='gpt-3.5-turbo', temperature=temperature, openai_api_key=os.getenv('OPENAI_API_KEY')),
            'claude-3-sonnet-20240229': lambda _: ChatAnthropic(model_name='claude-3-sonnet-20240229', temperature=temperature, anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')),
            'claude-3-opus-20240229': lambda _: ChatAnthropic(model_name='claude-3-opus-20240229', temperature=temperature, anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')),
        }

        self.chat_model = chat_model
               
    def chat(self, connection_string: str, query: str, chat_history: list[str] = []):
        """
        connection_string: str
        query: str
        chat_history: list of previous chat messages
        kwargs:
            namespace: str
            search_kwargs: dict
        """
        try: 

            # Connecting to the sqldb TODO:: pass connection_string
            connection_string=os.getenv('DB_CONNECTION_STRING')
            db = SQLDatabase.from_uri(connection_string)

            # validate the connection to the db
            # print(db.dialect)
            # print(db.get_usable_table_names())
            # response = db.run("SELECT TOP 10 * FROM T130Regioni")

            # Get GPT model
            # llm=self.llm_map[self.chat_model] 
            llm=ChatOpenAI(model='gpt-3.5-turbo', temperature=0.2, openai_api_key=os.getenv('OPENAI_API_KEY'))

            # ************************************************************
            # SQL Query chain 
            # ************************************************************
            system  = """You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run.
                    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data in the database.
                    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
                    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
                    Pay attention to use date('now') function to get the current date, if the question involves "today".

                    Only use the following tables:
                    {table_info}

                    Write an initial draft of the query. Then double check the {dialect} query for common mistakes, including:
                    - Using NOT IN with NULL values
                    - Using UNION when UNION ALL should have been used
                    - Using BETWEEN for exclusive ranges
                    - Data type mismatch in predicates
                    - Properly quoting identifiers
                    - Using the correct number of arguments for functions
                    - Casting to the correct data type
                    - Using the proper columns for joins

                    Use ALWAYS format:

                    First draft: <<FIRST_DRAFT_QUERY>>
                    Final answer: <<FINAL_ANSWER_QUERY>>
                    """

            prompt = ChatPromptTemplate.from_messages(
                [("system", system), ("human", "{input}")]
            ).partial(dialect=db.dialect)

            def parse_final_answer(output: str) -> str:
                print("**********************************************")
                print(output)
                print("**********************************************")
                if "Final answer: " in output:
                        return output.split("Final answer: ")[1]
                else:
                    # Gestione del caso in cui "Final answer" non è presente
                    print("No 'Final answer' found in the output.")             
                    print("**********************************************")
                    return output

            chain = create_sql_query_chain(llm, db, prompt=prompt) | parse_final_answer

            response = chain.invoke(
                        {
                            "question": query
                        }
                    )
            
            return response
        
            # ************************************************************
            # Agents 
            # ************************************************************

            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            tools = toolkit.get_tools()
            tools

            SQL_PREFIX = """You are an agent designed to interact with a SQL Server database.
            Given an input question, create a syntactically correct SQL Server query to run, then look at the results of the query and return the answer.
            Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
            You can order the results by a relevant column to return the most interesting examples in the database.
            Never query for all the columns from a specific table, only ask for the relevant columns given the question.
            You have access to tools for interacting with the database.
            Only use the below tools. Only use the information returned by the below tools to construct your final answer.
            You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

            To start you should ALWAYS look at the tables in the database to see what you can query.
            Do NOT skip this step.
            Then you should query the schema of the most relevant tables."""

            system_message = SystemMessage(content=SQL_PREFIX)

            agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)

            alert_info(f"Agent working on task..")

            response = agent_executor.invoke({"messages": [HumanMessage(content=f"{query}")]})

            alert_info(f"Task succesfully")

            # for s in agent_executor.stream(
            #     {"messages": [HumanMessage(content=f"{query}")]}
            # ):
            #     print(s)
            #     print("----")

            return response
        except Exception as e:
            alert_exception(e, "Error chatting")
            raise HTTPException(status_code=500, detail=f"Error chatting: {str(e)}")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens