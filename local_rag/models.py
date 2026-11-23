from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SourceDocument:
    text: str
    title: str
    source_path: str
    source_type: str
    locator: str = ""
    source_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    title: str
    source_path: str
    source_type: str
    locator: str = ""
    source_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Chunk":
        return cls(**value)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    semantic_score: float
    lexical_score: float


@dataclass
class Answer:
    text: str
    results: list[SearchResult]
    refused: bool = False
    sql_used: str = ""


@dataclass
class IndexUpdateResult:
    status: str
    new_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    unchanged_files: list[str] = field(default_factory=list)
    failed_files: dict[str, str] = field(default_factory=dict)
    new_embeddings: int = 0
    reused_embeddings: int = 0
    total_files: int = 0
    total_chunks: int = 0
    elapsed_seconds: float = 0.0
    manifest: Any = None
    errors: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


