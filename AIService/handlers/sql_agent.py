from langchain_core.language_models import LanguageModelLike
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.callbacks.manager import get_openai_callback

# ************************************************************
# create_sql_agent - [fail]
# ************************************************************           
def sql_agent(user_question: str, db: SQLDatabase, llm: LanguageModelLike):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    context = toolkit.get_context() 
    tools = toolkit.get_tools()
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True, tools=tools, context=context)
    with get_openai_callback() as cb:
        response=agent_executor.invoke({user_question})

    print("**********************************************")
    print(cb)

    # Extract relevant information from cb
    openai_fee = {
        "completion_tokens": cb.completion_tokens,  
        "prompt_tokens": cb.prompt_tokens ,
        "total_cost": cb.total_cost
    }

    return {
        "model_response": response,
        "model_fee": openai_fee
    }