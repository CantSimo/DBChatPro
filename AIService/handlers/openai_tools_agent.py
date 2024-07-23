from langchain_core.language_models import LanguageModelLike
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.prompts.chat import (
    ChatPromptTemplate, 
    MessagesPlaceholder,
    PromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage)
from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX
from langchain.agents import (create_openai_tools_agent, agent_types)
from langchain.agents.agent import AgentExecutor
from langchain_community.callbacks.manager import get_openai_callback

# ************************************************************
# create_openai_tools_agent - [failed]
# ************************************************************           
def openai_tools_agent(user_question: str, db: SQLDatabase, llm: LanguageModelLike):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    context = toolkit.get_context()
    tools = toolkit.get_tools()

    messages = [
        HumanMessagePromptTemplate.from_template("{input}"),
        AIMessage(content=SQL_FUNCTIONS_SUFFIX),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    prompt = prompt.partial(**context)            
    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=toolkit.get_tools(),
        verbose=True,
    )

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
