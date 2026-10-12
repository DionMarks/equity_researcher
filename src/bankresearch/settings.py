"""Runtime settings. Secrets come from the environment and are never logged."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _repo_root() -> Path:
    """The repository root, found from this file rather than the cwd."""
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # utf-8-sig reads a .env saved by Windows tools that add a byte-order mark, which would
    # otherwise hide the first key.
    model_config = SettingsConfigDict(
        env_prefix="BANKRESEARCH_", env_file=".env", env_file_encoding="utf-8-sig", extra="ignore"
    )

    config_dir: Path = Field(default_factory=lambda: _repo_root() / "config")
    data_dir: Path = Field(default_factory=lambda: _repo_root() / "data")
    runs_dir: Path = Field(default_factory=lambda: _repo_root() / "runs")
    evals_dir: Path = Field(default_factory=lambda: _repo_root() / "evals")

    # Read from ANTHROPIC_API_KEY (no prefix) and never written to a log, a trace or a
    # result. SecretStr keeps it out of the model's repr, so if the settings object ever
    # ends up in a log line or a traceback the key shows as **********. Only the client
    # calls get_secret_value().
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db" / "research.sqlite"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_api_key.get_secret_value())


def get_settings() -> Settings:
    return Settings()
