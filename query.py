from llama_index.core.query_engine import CustomQueryEngine 
from llama_index.core.llms import LLM 
from llama_index.core import PropertyGraphIndex 
from store import *

import re

class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore 
    index: PropertyGraphIndex
    llm:LLM 
    similarity_top_k: int = 20 

    def custom_query(self, query_str: str) -> str:
        entities = self.get_entities(query_str, self.similarity_top_k)

        community_ids = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for id, community_summary in community_summaries.items()
            if id in community_ids
        ]

        final_answer = self.aggregate_answers(community_answers)
        return final_answer


    def get_entities(self, query_str, similarity_top_k):
        nodes_retrieved = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        ).retrieve(query_str)

        entities = set()
        pattern = (
            r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"
        )

        for node in nodes_retrieved:
            matches = re.findall(
                pattern, node.text, re.MULTILINE | re.IGNORECASE
            )
            for match in matches:
                subject = match[0]
                obj = match[2]
                entities.add(subject)
                entities.add(obj)

        return list(entities) 

    def retrieve_entity_communities(self, entity_info, entities):
        community_ids = []

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])

        return list(set(community_ids))

    def generate_answer_from_summary(self, community_summary, query):
        prompt = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [
            ChatMessage(role = "system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above information.",
            )
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(
            r"^assistant:\s*", "", str(response)
        ).strip()
        return cleaned_response


    def aggregate_answers(self, community_answers):
        prompt = "Combine the following intermediate answers into a final, concise response."
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",
            ),
        ]
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response