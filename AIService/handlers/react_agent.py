from langchain_core.language_models import LanguageModelLike
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage)
from langgraph.prebuilt import create_react_agent
from langchain_community.callbacks.manager import get_openai_callback

# ************************************************************
# create_react_agent - [OK]
# ************************************************************
def react_agent(user_question: str, db: SQLDatabase, llm: LanguageModelLike):

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    SQL_PREFIX = """You are an agent designed to interact with a SQL Server database.
    Given an input question, create a syntactically correct SQL Server query to run, then look at the results of the query and return the answer.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables.
    ALWAYS return the final query you used to give the response.
    Use ALWAYS format:

    Query: <<FINAL_QUERY>>
    Final answer: <<FINAL_ANSWER>>
    """
    system_message = SystemMessage(content=SQL_PREFIX)

    agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)

    with get_openai_callback() as cb:
        response = agent_executor.invoke({"messages": [HumanMessage(content=f"{user_question}")]})

    print("**********************************************")
    print(cb)

    # for s in agent_executor.stream(
    #     {"messages": [HumanMessage(content=f"{user_question}")]}
    # ):
    #     print(s)
    #     print("----")

    # Extract relevant information from cb
    openai_fee = {
        "completion_tokens": cb.completion_tokens,  
        "prompt_tokens": cb.prompt_tokens ,
        "total_cost": cb.total_cost
    }

    # get last message
    last_message = response['messages'][-1]

    query, risposta_finale=estrai_query_e_risposta(last_message.content)

    return {
        "model_response": query,
        "model_verbal_response": risposta_finale,
        "model_fee": openai_fee
    }

def estrai_query_e_risposta(testo):
    # Trova l'indice di inizio e fine della query
    inizio_query = testo.find('Query:') + len('Query:')
    fine_query = testo.find('Final answer:')

    # Estrai la query
    query = testo[inizio_query:fine_query].strip()

    # Estrai la risposta finale
    risposta_finale = testo[fine_query + len('Final answer:'):].strip()

    # Restituisci la query e la risposta finale
    return query, risposta_finale
