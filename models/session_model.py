"""
Data model for Session entity.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """Represents a single query-response exchange."""
    query: str
    response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class Session(BaseModel):
    """Represents a user's conversation context."""
    id: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    history: List[ConversationTurn] = Field(default_factory=list, max_items=50)
    context: Dict[str, Any] = Field(default_factory=dict)

    def add_turn(self, query: str, response: str, sources: List[Dict[str, Any]] = None) -> None:
        """Add a conversation turn to history."""
        turn = ConversationTurn(
            query=query,
            response=response,
            sources=sources or []
        )
        self.history.append(turn)
        self.last_active = datetime.utcnow()

        # Keep only last 50 exchanges
        if len(self.history) > 50:
            self.history = self.history[-50:]

    def is_expired(self, hours: int = 24) -> bool:
        """Check if session has expired."""
        time_diff = datetime.utcnow() - self.last_active
        return time_diff.total_seconds() > (hours * 3600)
