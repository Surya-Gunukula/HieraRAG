import json
import os
from pydantic import BaseModel, Field
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate

# ——— Setup LLM ———
api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

# ——— Define Structured Output Models ———
class RatedAnswer(BaseModel):
    comprehensiveness_rating: int = Field(ge=0, le=100, description="Detail coverage score.")
    comprehensiveness_description: str = Field(description="Why that score?")

    diversity_rating: int = Field(ge=0, le=100, description="Variety of perspectives score.")
    diversity_description: str = Field(description="Why that score?")

    empowerment_rating: int = Field(ge=0, le=100, description="Usefulness to the reader score.")
    empowerment_description: str = Field(description="Why that score?")

    directness_rating: int = Field(ge=0, le=100, description="Conciseness score.")
    directness_description: str = Field(description="Why that score?")

class Evaluation(BaseModel):
    evaluation: RatedAnswer

structured_llm = llm.with_structured_output(Evaluation)

# ——— Craft the Prompt ———
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are an expert judge.  
Rate the following answer from 0–100 on each category:
1) Comprehensiveness: How much detail does it cover?  
2) Diversity: How varied/rich are its perspectives?  
3) Empowerment: How well does it help the reader make informed judgments?  
4) Directness: How concise is it?

Explain each score in 1–2 sentences.  
Respond **only** with JSON matching the Evaluation model."""
    ),
    ("human", "Question: {question}\nAnswer: {answer}")
])

# ——— Load & Evaluate ———
scores: Dict[str, float] = {
    "comprehensiveness": 0,
    "diversity": 0,
    "empowerment": 0,
    "directness": 0,
}
count = 0

with open("results_my_llm.json") as f:
    responses = json.load(f)["responses"]

for question, answer in responses.items():
    msgs = prompt.format_messages(question=question, answer=answer)
    out = structured_llm.invoke(msgs)
    ev = out.evaluation

    scores["comprehensiveness"] += ev.comprehensiveness_rating
    scores["diversity"]       += ev.diversity_rating
    scores["empowerment"]     += ev.empowerment_rating
    scores["directness"]      += ev.directness_rating
    count += 1

# ——— Print Averages ———
for metric in scores:
    avg = scores[metric] / count
    print(f"{metric.title()} avg: {avg:.1f}")
