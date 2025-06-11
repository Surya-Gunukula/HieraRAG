import os
import pandas as pd
from typing_extensions import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph.base import PropertyGraphIndex

from query import *


# --- Setup LLM & GraphRAG backend ---
api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
index_llm = OpenAI(model="gpt-4o", api_key=api_key)

def build_graph_index(corpus_docs):
    # Split into nodes
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(corpus_docs)
    # Extract graph
    graph_store = Neo4jPropertyGraphStore(
        username=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASS", "changeme"),
        url=os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
    )
    index = PropertyGraphIndex(
        nodes=nodes,
        kg_extractors=[],  # assume extractor injected if needed
        property_graph_store=graph_store,
        show_progress=False,
    )
    index.property_graph_store.build_communities()
    return index, graph_store

# Example: load documents
news = pd.read_csv(...)
documents = [Document(text=f"{r['title']}: {r['text']}") for _, r in news.iterrows()]
index, graph_store = build_graph_index(documents)
query_engine = GraphRAGQueryEngine(graph_store=index.property_graph_store, llm=index_llm, index=index)

# Simple in-memory cache
CACHE = {}

# --- Define State Schema ---
class State(TypedDict):
    input: str
    decision: Literal["cache", "agents"]
    agent1_ctx: str
    agent2_ctx: str
    community_answer: str
    critique: str
    accept: bool
    cached_answer: str
    final_answer: str

# --- Node Functions ---
class DecisionModel(BaseModel):
    decision: Literal["cache", "agents"] = Field(
        description="Choose 'cache' if you have a stored response, else 'agents' to trigger graph agents."
    )

decider = llm.with_structured_output(DecisionModel)

def orchestrator(state: State):
    # check cache
    ans = CACHE.get(state['input'])
    if ans:
        return {"decision": "cache", "cached_answer": ans}
    # ask LLM whether to use cache or agents
    out = decider.invoke([
        SystemMessage(content="Decide: use 'cache' if you can immediately answer from memory, else 'agents'."),
        HumanMessage(content=f"Query: {state['input']}")
    ])
    return {"decision": out.decision}

# Graph Agent implementations
# Assume query_engine available in global scope

def graph_agent_1(state: State):
    nodes = query_engine.graph_store.query_communities(state['input'], level=1)
    ctx = "\n".join([n.get_content() for n in nodes])
    return {"agent1_ctx": ctx}

def graph_agent_2(state: State):
    nodes = query_engine.graph_store.query_communities(state['input'], level=2)
    ctx = "\n".join([n.get_content() for n in nodes])
    return {"agent2_ctx": ctx}

# Synthesizer

def synthesizer(state: State):
    combined = state['agent1_ctx'] + "\n" + state['agent2_ctx']
    resp = llm.invoke([
        SystemMessage(content="Synthesize a concise community answer from contexts."),
        HumanMessage(content=f"Query: {state['input']}\nContext: {combined}")
    ])
    return {"community_answer": resp.content}

# Critiquer
class CritiqueModel(BaseModel):
    accept: bool = Field(description="True if context is relevant and sufficient.")
    critique: str = Field(description="Feedback on context quality.")

critiquer = llm.with_structured_output(CritiqueModel)

def critiquer_node(state: State):
    out = critiquer.invoke([
        SystemMessage(content="Evaluate if the community answer is relevant and sufficient. Return accept=True/False."),
        HumanMessage(content=f"Answer: {state['community_answer']}")
    ])
    return {"accept": out.accept, "critique": out.critique}

# Final Answer

def final_answer(state: State):
    if state['decision'] == 'cache':
        return {'final_answer': state['cached_answer']}
    resp = llm.invoke([
        SystemMessage(content="Generate the final answer based on community_answer."),
        HumanMessage(content=state['community_answer'])
    ])
    # update cache
    CACHE[state['input']] = resp.content
    return {'final_answer': resp.content}

# --- Build StateGraph ---
router = StateGraph(State)
router.add_node("orchestrator", orchestrator)
router.add_node("graph_agent_1", graph_agent_1)
router.add_node("graph_agent_2", graph_agent_2)
router.add_node("synthesizer", synthesizer)
router.add_node("critiquer", critiquer_node)
router.add_node("final_answer", final_answer)

router.add_conditional_edges(
    START,
    lambda s: s['decision'] if 'decision' in s else orchestrator(s)['decision'],
    {"cache": "final_answer", "agents": "graph_agent_1"}
)
router.add_edge("graph_agent_1", "graph_agent_2")
router.add_edge("graph_agent_2", "synthesizer")
router.add_edge("synthesizer", "critiquer")
router.add_conditional_edges(
    "critiquer",
    lambda s: 'final_answer' if s['accept'] else 'synthesizer',
    {True: "final_answer", False: "synthesizer"}
)
router.add_edge("final_answer", END)

workflow = router.compile()

# Example invocation
# result = workflow.invoke({"input": "What are the main news discussed?"})
# print(result['final_answer'])