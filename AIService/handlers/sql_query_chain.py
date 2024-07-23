import re
from langchain_core.prompts.chat import (
    ChatPromptTemplate, 
    MessagesPlaceholder,
    PromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.language_models import LanguageModelLike
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage)
from langgraph.prebuilt import create_react_agent
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains import (
    LLMChain, 
    ConversationalRetrievalChain, 
    create_history_aware_retriever, 
    create_retrieval_chain, 
    create_sql_query_chain)

# ************************************************************
# create_sql_query_chain - [OK]
# ************************************************************
def sql_query_chain(user_question: str, db: SQLDatabase, llm: LanguageModelLike):
    # Prompt for gpt-3.5-turbo    
    system  = """You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run.
            Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the TOP clause as per {dialect}. You can order the results to return the most informative data in the database.
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
            
            Question: {input}
            
            Use ALWAYS format:

            First draft: <<FIRST_DRAFT_QUERY>>
            Final answer: <<FINAL_ANSWER_QUERY>>
            """

    # Prompt for gpt-4o-mini
    # system  = '''You are a {dialect} expert. Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    #         Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the TOP clause as per {dialect}. You can order the results to return the most informative data in the database.
    #         Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
    #         Pay attention to use only the column names you can see in the tables below. BE CAREFUL to not query for columns that do not exist. Also, pay attention to which column is in which table.
    #         Pay attention to use date('now') function to get the current date, if the question involves "today".

    #         Only use the following tables:
    #         {table_info}

    #         Write an initial draft of the query. Then double check the {dialect} query for common mistakes, including:
    #         - Using NOT IN with NULL values
    #         - Using UNION when UNION ALL should have been used
    #         - Using BETWEEN for exclusive ranges
    #         - Data type mismatch in predicates
    #         - Properly quoting identifiers
    #         - Using the correct number of arguments for functions
    #         - Casting to the correct data type
    #         - Using the proper columns for joins
    #         - double check columns name with database schema
            
    #         Use the following format:

    #         Question: "Question here"
    #         SQLQuery: "SQL Query to run"
    #         SQLResult: "Result of the SQLQuery"
    #         Answer: "Final answer here"

    #         Question: {input}'''

    prompt = ChatPromptTemplate.from_messages(
        [("system", system)]
    ).partial(dialect=db.dialect)

    # Parser for gpt-3.5-turbo
    def parse_final_answer(output: str) -> str:
        print("**********************************************")
        print(output)
        print("**********************************************")
        if "Final answer: " in output:
                return output.split("Final answer: ")[1]
        else:
            if "First draft: " in output:
                    return output.split("First draft: ")[1]
            else:
                # Gestione del caso in cui "Final answer" non è presente
                print("No 'Final answer' found in the output.")             
                print("**********************************************")
                return output
            
    # Parser for gpt-4o-mini
    def extract_sql_query(input_string: str) -> str:
        # Regex pattern to match the SQL query within triple backticks
        pattern = r'SQLQuery:\s*```sql\n(.*?)\n```'
        
        # Search for the pattern in the input string
        match = re.search(pattern, input_string, re.DOTALL)
        
        if match:
            # Return the cleaned SQL query
            return match.group(1).strip()
        else:
            return None

    chain = create_sql_query_chain(llm, db, prompt=prompt) | parse_final_answer
    # chain = create_sql_query_chain(llm, db, prompt=prompt) | extract_sql_query

    #chain.get_prompts()[0].pretty_print()
    #print("**********************************************")

    with get_openai_callback() as cb:
        response = chain.invoke(
                    {
                        "question": user_question
                    }
                )
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