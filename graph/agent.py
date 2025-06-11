from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.chat_models import ChatOpenAI 
from langchain.tools.base import Tool
from neo4j import GraphDatabase
import json 
import random
import numpy as np
import os

from GraphRAG.graph.main import *


def graphrag_tool_func(query: str)-> str:
    return query_engine.custom_query(query)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "S00stest!"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

graphrag_tool = Tool(
    name = "GraphRAG",
    func = graphrag_tool_func,
    description = "Use this tool for questions that require multi-hop reasoning or involve relationships"
)

def run_langchain_agent(user_query: str):

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(model_name="gpt-4o", temperature = 0, openai_api_key=api_key)

    agent = initialize_agent(
        tools = [graphrag_tool],
        llm = llm,
        agent = AgentType.OPENAI_FUNCTIONS,
        verbose = True
    )

    return agent.run(user_query)


    
if __name__ == "__main__":
    
    query = "What are the main news in energy sector?"
    result = run_langchain_agent(query)
    print(result)

