from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # OpenAI Settings
    API_KEY: str
    MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # Changed this line!
    
    # Document Processing Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Storage Settings
    VECTOR_DB_PATH: Path = Path("./data/chroma_db")
    UPLOAD_DIR: Path = Path("./uploads")
    
    # LLM Settings
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.7
    SYSTEM_PROMPT: str = "You are a helpful AI assistant. Answer questions based on the provided context."
    
    # API Settings
    HOST: str = "localhost"
    PORT: int = 8000
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

# Create settings instance
settings = Settings()