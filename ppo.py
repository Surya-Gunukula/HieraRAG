import os
import gym
import numpy as np
from stable_baselines3 import PPO
from typing import List, Optional
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph.base import PropertyGraphIndex


class GraphRLEnv(gym.Env):
    """
    A Gym environment that implements the Graph-R1 PPO training loop:
    Query → Policy LLM (Graph Agent) → Node retrieval → Synthesizer → Answer → Reward LLM → Policy update
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        documents: List[Document],
        query: str,
        llm_model: str = "gpt-4o",
        reward_model: str = "gpt-4o",
        max_steps: int = 5,
        top_k: int = 10,
    ):
        super().__init__()
        # Store query and LLMs
        self.query = query
        api_key = os.environ["OPENAI_API_KEY"]
        self.agent_llm = OpenAI(model=llm_model, api_key=api_key)
        self.synth_llm = OpenAI(model=llm_model, api_key=api_key)
        self.reward_llm = OpenAI(model=reward_model, api_key=api_key)

        # Build graph index
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        nodes = splitter.get_nodes_from_documents(documents)
        self.graph_store = Neo4jPropertyGraphStore(
            username=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASS", "changeme"),
            url=os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
        )
        self.index = PropertyGraphIndex(
            nodes=nodes,
            kg_extractors=[],  # your extractor if needed
            property_graph_store=self.graph_store,
        )
        self.index.property_graph_store.build_communities()

        # Define action & observation spaces
        # Actions: choose one of top_k neighbors or STOP
        self.top_k = top_k
        self.action_space = gym.spaces.Discrete(self.top_k + 1)

        # Observations: dummy vector (e.g., query embedding)
        # Here we use a fixed-size embedding
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32
        )

        self.max_steps = max_steps
        self.reset()

    def reset(self):
        # Start episode: clear path, embed query
        self.path: List[str] = []
        self.steps = 0
        self.done = False
        # Pre-compute query embedding as observation
        self.query_embed = self.agent_llm.embed_query(self.query)
        return self.query_embed

    def step(self, action: int):
        reward = 0.0
        info = {}

        if action == self.top_k:
            # STOP action → synthesize and reward
            community_ctx = self._collect_context()
            answer = self._synthesize(community_ctx)
            reward = self._compute_reward(answer)
            self.done = True
            obs = self.query_embed
        else:
            # Traverse to one of the top-k neighbors
            neighbors = self.graph_store.get_topk_neighbors(self.query, k=self.top_k)
            node = neighbors[action]
            self.path.append(node.id)
            obs = self.query_embed  # still same observation
            self.steps += 1
            if self.steps >= self.max_steps:
                self.done = True

        return obs, reward, self.done, info

    def _collect_context(self) -> str:
        # Gather node contents along the path
        contents = []
        for node_id in self.path:
            node = self.graph_store.get_node(node_id)
            contents.append(node.get_content())
        return "\n".join(contents)

    def _synthesize(self, context: str) -> str:
        prompt = f"Query: {self.query}\nContext:\n{context}\nProvide a concise answer."  
        resp = self.synth_llm.chat(prompt)
        return resp

    def _compute_reward(self, answer: str) -> float:
        # Reward model judging answer relevance
        prompt = f"Journalist: rate the following answer on relevance to query '{self.query}' from 0 to 1.\nAnswer: {answer}"
        score = self.reward_llm.chat(prompt)
        try:
            return float(score)
        except:
            return 0.0


if __name__ == "__main__":
    # Example usage
    # Load your corpus
    docs = [Document(text="Sample document about AI."), Document(text="Another knowledge snippet.")]
    env = GraphRLEnv(docs, query="What is the main theme of the corpus?")

    # Validate env
    from stable_baselines3.common.env_checker import check_env
    check_env(env)

    # Train with PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Save and test
    model.save("graph_agent_ppo")
    obs = env.reset()
    action, _ = model.predict(obs)
    print("Chosen action:", action)