"""Local RAG prototype."""

from .config import Settings
from .models import IndexUpdateResult
from .service import RAGService
from .table_parsing import TABLE_PARSER_VERSION

__all__ = ["Settings", "RAGService", "IndexUpdateResult", "TABLE_PARSER_VERSION"]

