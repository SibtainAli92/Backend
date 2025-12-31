"""
Data models for Query, Response, and chat-related entities.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Query(BaseModel):
    """Represents a user's question with associated metadata."""
    id: str
    text: str
    mode: str = Field(..., pattern="^(rag|selected)$")
    session_id: str
    selected_text: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Source(BaseModel):
    """Represents a source citation."""
    text: str
    book_id: str
    chapter_id: Optional[str] = None
    section_id: Optional[str] = None
    source_url: Optional[str] = None
    score: Optional[float] = None


class Response(BaseModel):
    """Represents the agent's answer with citations."""
    id: str
    text: str
    query_id: str
    session_id: str
    sources: List[Source] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Model for chat endpoint requests."""
    message: str = Field(..., min_length=1)
    selected_text: Optional[str] = Field(None, max_length=10000)
    session_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    """Model for chat endpoint responses."""
    response: str
    session_id: str
    sources: List[Source] = Field(default_factory=list)
    error: Optional[str] = None


class RAGToolRequest(BaseModel):
    """Model for internal RAG tool requests."""
    query: str
    mode: str = Field(..., pattern="^(rag|selected)$")
    top_k: int = Field(default=5, ge=1, le=20)
    selected_text: Optional[str] = None


class RAGToolResponse(BaseModel):
    """Model for internal RAG tool responses."""
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
