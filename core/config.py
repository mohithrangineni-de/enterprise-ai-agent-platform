"""
core/config.py

Centralised settings loaded from environment variables / .env file.
"""

from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = "gpt-4o"

    # Database
    db_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/analytics"
    db_name: str = "analytics"

    # Vector store
    vector_store_path: str = "./data/faiss_index"
    pinecone_api_key: str = ""
    pinecone_index: str = ""
    rag_top_k: int = 6

    # Memory
    short_term_window: int = 10  # conversation turns to keep

    # Observability
    log_level: str = "INFO"
    json_logs: bool = True
    otlp_endpoint: str = ""  # e.g. http://localhost:4317

    # App
    api_port: int = 8000


settings = Settings()
