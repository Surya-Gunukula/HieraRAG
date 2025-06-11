import pandas as pd
from llama_index.core import Document 

from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from GraphRAG.graph.extractor import *
from GraphRAG.graph.query import *
from GraphRAG.graph.store import *
from utils import clear_neo4j_db 

import os 
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore 
from llama_index.core import PropertyGraphIndex
from llama_index.core.node_parser import SentenceSplitter

from neo4j import GraphDatabase

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.chat_models import ChatOpenAI 
from langchain.tools.base import Tool
import json 
import random
import numpy as np


def graphrag_tool_func(query: str)-> str:
    return query_engine.custom_query(query)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "S00stest!"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

graphrag_tool = Tool(
    name = "GraphRAG",
    func = graphrag_tool_func,
    description = "Use this tool for questions that require multi-hop reasoning or involve relationships, always attempt to answer the question over dodging and recommending other sources."
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

    clear_neo4j_db("bolt://localhost:7687", "neo4j", "S00stest!")

    news = pd.read_csv(
    "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
    )[:3]
    news.head()

    documents = [
        Document(text=f"{row['title']}: {row['text']}")
        for i, row in news.iterrows()
    ]
    print(documents[0])

    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    llm = OpenAI(model = "gpt4", api_key=api_key)
    Settings.llm = OpenAI(api_key=api_key, model="gpt-4")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    """

    llm = Ollama(
    model = "llama3",
    base_url = "http://localhost:11434",
    request_timeout = 120.0
    )
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name = "BAAI/bge-base-en-v1.5")

    # Set up extractor
    extractor = GraphRAGExtractor(
        llm=Settings.llm,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        parse_fn=parse_fn,
        max_paths_per_chunk=10,
    )

    graph_store = GraphRAGStore(
        username="neo4j", password="S00stest!", url="bolt://localhost:7687"
    )

    # Run extraction
    processed_docs = extractor(documents)

    print(processed_docs[0].metadata.get(KG_NODES_KEY))

    for doc in processed_docs:
        print(f"\nDoc ID: {doc.id_}")
        print(f"Text: {doc.text[:300]}...")  # truncate for readability
        print("📌 Entities:")
        for node in doc.metadata.get(KG_NODES_KEY, []):
            print(f"  - {node.name} ({node.label}) - {node.properties.get('entity_description')}")
        print("🔗 Relationships:")
        for rel in doc.metadata.get(KG_RELATIONS_KEY, []):
            print(f"  - {rel.source_id} --[{rel.label}]--> {rel.target_id}")
            print(f"    Explanation: {rel.properties.get('relationship_description')}")
        print("=" * 60)

    splitter = SentenceSplitter(
        chunk_size = 1024,
        chunk_overlap = 20,
    )

    nodes = splitter.get_nodes_from_documents(documents)


    index = PropertyGraphIndex(
        nodes=nodes, 
        kg_extractor=[extractor],
        property_graph_store = graph_store,
        show_progress = False
    )

    print(index.property_graph_store.get_triplets()[10])
    print(index.property_graph_store.get_triplets()[10][0].properties)
    print(index.property_graph_store.get_triplets()[10][1].properties)

    print("GOT HERE")

    index.property_graph_store.build_communities()


    query_engine = GraphRAGQueryEngine(
        graph_store=index.property_graph_store,
        llm=llm,
        index=index,
        similarity_top_k=10,
    )

    response = query_engine.custom_query2(
        "What are the main news discussed in the document?"
    )
    print(response)


    print("SEPERATION")

    response = query_engine.custom_query2("What are the main news in energy sector?")
    print(response)

    print("SEPERATION")

    query = "What are the main news in energy sector?"
    result = run_langchain_agent(query)
    print(result)




    

