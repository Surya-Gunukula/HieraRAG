import pandas as pd
from llama_index.core import Document
import os
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex


from extractor import * 
from extract_template import *
from query import * 
from store import * 
from utils import *




news = pd.read_csv(
    "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
)[:50]

news.head()


documents = [
    Document(text=f"{row['title']}: {row['text']}")
    for i, row in news.iterrows()
]

api_key = os.environ.get("OPENAI_API_KEY")


llm = OpenAI(model="gpt-4o")


splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,
)
nodes = splitter.get_nodes_from_documents(documents)

kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    max_paths_per_chunk=2,
    parse_fn=parse_fn,
)

graph_store = GraphRAGStore(
    username="neo4j", password="S00stest!", url="bolt://localhost:7687"
)

index = PropertyGraphIndex(
    nodes=nodes,
    kg_extractors=[kg_extractor],
    property_graph_store=graph_store,
    show_progress=True,
)

index.property_graph_store.build_communities()

query_engine = GraphRAGQueryEngine(
    graph_store=index.property_graph_store,
    llm=llm,
    index=index,
    similarity_top_k=10,
)

response = query_engine.query(
    "What are the main news discussed in the document?"
)

print(response)
