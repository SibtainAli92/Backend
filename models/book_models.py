"""
Data models for Book, Chapter, Section, and Chunk entities.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Chapter(BaseModel):
    """Represents a chapter within the robotics book."""
    id: str
    title: str
    content: str
    section_ids: List[str] = Field(default_factory=list)
    book_id: str


class Section(BaseModel):
    """Represents a section within a chapter."""
    id: str
    title: str
    content: str
    chapter_id: str
    book_id: str


class BookContent(BaseModel):
    """Represents the robotics book content."""
    id: str
    title: str
    content: str
    chapters: List[Chapter] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_modified: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    """Represents a chunk of book content with embedding."""
    id: str
    text: str
    book_id: str
    chapter_id: Optional[str] = None
    section_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
