from pydantic import BaseModel
from typing import List, Tuple

class ChatRequest(BaseModel):
    query: str
    history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    answer: str
    history: List[Tuple[str, str]]