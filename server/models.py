"""
Agent2Memory (A2M) Protocol
Pydantic models matching the wire format defined in §6 of the spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    working    = "working"
    episodic   = "episodic"
    semantic   = "semantic"
    procedural = "procedural"
    external   = "external"


class EntryMeta(BaseModel):
    source_agent:     Optional[str]   = None
    source_framework: Optional[str]   = None
    created_at:       str             = ""   # ISO8601, server-set
    updated_at:       str             = ""   # ISO8601, server-set
    ttl_s:            Optional[int]   = None # seconds; None = permanent
    tags:             list[str]       = Field(default_factory=list)
    confidence:       Optional[float] = None


class EntryWrite(BaseModel):
    """Write / upsert request body."""
    key:       str
    type:      MemoryType
    value:     Any
    embedding: Optional[list[float]] = None  # caller-provided; required for semantic query
    meta:      EntryMeta             = Field(default_factory=EntryMeta)


class Entry(BaseModel):
    """Full entry returned by read / list / query."""
    id:        str
    key:       str
    namespace: str
    type:      MemoryType
    value:     Any
    embedding: Optional[list[float]] = None
    meta:      EntryMeta


class QueryRequest(BaseModel):
    """Semantic query request body.
    Caller MUST provide `embedding` — A2M never generates embeddings (spec §5.1).
    """
    embedding:  list[float]           # required — caller embeds the query text
    type:       Optional[MemoryType]  = None
    top_k:      int                   = 5
    min_score:  Optional[float]       = None
    tag:        list[str]             = Field(default_factory=list)
    recursive:  bool                  = False


class QueryResult(BaseModel):
    entry: Entry
    score: float


class ListResponse(BaseModel):
    entries:     list[Entry]
    total:       int
    next_offset: Optional[int]


class ErrorBody(BaseModel):
    code:    str
    message: str
