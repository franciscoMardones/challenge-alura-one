"""Shared configuration module for BimBam Buy RAG Agent."""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")

# Paths
CHROMA_DIR: str = "chroma_db"
DOCS_DIR: str = "docs"

# Model settings
EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL: str = "openai/gpt-oss-120b"

# Chunking
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100

# Retrieval
TOP_K: int = 6
SIMILARITY_THRESHOLD: float = 17.3  # L2 distance; empirical P75 of in-scope scores
MMR_LAMBDA: float = 0.7
FALLBACK_MESSAGE: str = (
    "No encontré información relevante en los documentos "
    "de BimBam Buy para responder tu pregunta."
)

# Generation
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 800
