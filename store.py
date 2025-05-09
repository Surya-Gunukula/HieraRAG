import re 
import networkx as nx 
from graspologic.partition import hierarchical_leiden
from collections import defaultdict 

from llama_index.core.llms import ChatMessage
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama


class GraphRAGStore(Neo4jPropertyGraphStore):
    community_summary = {}
    entity_info = None
    max_cluster_size = 5
    
    llm = Ollama(
    model = "llama3",
    base_url = "http://localhost:11434",
    request_timeout = 120.0
    )

    def generate_community_summary(self, text):
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
                
            ),
            ChatMessage(role="user", content=text),
        ]
        response = self.llm.chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        print(clean_response)
        return clean_response

    def build_communities(self):
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        self.entity_info, community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        print(triplets)
        triplets = [triple for triple in triplets if triple[1].properties.get("relationship_description") is not None]
        print(triplets)
        for entity1, relation, entity2 in triplets:
            print("PROPERTIES", relation.properties)
            print("RELATION PROPERTIES", relation.properties.get("relationship_description"))
            print(entity1, entity2)
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            nx_graph.add_edge(
                relation.source_id, 
                relation.target_id,
                relationship = relation.label,
                description = relation.properties["relationship_description"]
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node = item.node
            cluster_id = item.cluster

            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(detail)

        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info)

    def _summarize_communities(self, community_info):
        for community_id, details in community_info.items():
            details_text = (
                "\n".join(details) + "."
            )
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        if not self.community_summary:
            self.build_communities()
        return self.community_summary 