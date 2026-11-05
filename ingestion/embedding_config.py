"""Shared FastEmbed model configuration for indexing and retrieval."""

from pathlib import Path

from fastembed import TextEmbedding


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_CACHE_DIR = Path(__file__).resolve().parent.parent / "models_cache"


def create_embedding_model(*, allow_download: bool = False) -> TextEmbedding:
    """Load the configured model from the repository cache without downloading."""
    return TextEmbedding(
        model_name=EMBEDDING_MODEL,
        cache_dir=str(EMBEDDING_CACHE_DIR),
        local_files_only=not allow_download,
    )
