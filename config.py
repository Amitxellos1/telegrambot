"""
Configuration module for the Telegram RAG Bot.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Application configuration."""
    
    # Telegram Bot
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    
    # LLM Configuration
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "mistral"))
    ollama_vision_model: str = field(default_factory=lambda: os.getenv("OLLAMA_VISION_MODEL", "llava"))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    
    # Embedding Configuration
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    
    # RAG Configuration
    data_directory: str = field(default_factory=lambda: os.getenv("DATA_DIRECTORY", "data"))
    db_directory: str = field(default_factory=lambda: os.getenv("DB_DIRECTORY", "db"))
    collection_name: str = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "knowledge_base"))
    top_k_results: int = field(default_factory=lambda: int(os.getenv("TOP_K_RESULTS", "3")))
    
    # Message History
    max_history_per_user: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY_PER_USER", "3")))
    
    def validate(self):
        """Validate required configuration."""
        if not self.telegram_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required. Set it in .env file.")
        
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI. Set it in .env file.")
        
        return True


# Global config instance
config = Config()



