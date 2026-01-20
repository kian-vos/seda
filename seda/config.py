"""Configuration management using pydantic-settings."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Keys
    apify_api_token: str = ""
    anthropic_api_key: str = ""

    # Database - Local SQLite
    database_path: str = "data/seda.db"

    # Database - Turso (cloud)
    turso_database_url: str = ""
    turso_auth_token: str = ""

    # Directories
    data_dir: str = "data"
    models_dir: str = "data/models"
    cache_dir: str = "data/cache"

    # Scraper settings
    scraper_batch_size: int = 10
    scraper_max_tweets_per_user: int = 100
    scraper_max_retweeters: int = 50
    scraper_rate_limit_delay: float = 1.0

    # Analysis settings
    bot_threshold: float = 0.7
    coordination_time_window_minutes: int = 5
    minhash_threshold: float = 0.7
    minhash_num_perm: int = 128

    # LLM settings
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 1024
    llm_batch_size: int = 10

    # Logging
    log_level: str = "INFO"

    @property
    def db_path(self) -> Path:
        return Path(self.database_path)

    @property
    def use_turso(self) -> bool:
        """Check if Turso cloud database is configured."""
        return bool(self.turso_database_url and self.turso_auth_token)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir)

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)

    def ensure_dirs(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
