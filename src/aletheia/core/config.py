"""Typed application configuration.

Values come from ``ALETHEIA_``-prefixed environment variables and, when present, a local ``.env``.
Production refuses to start with a demo-shaped configuration: a missing database URL or signing key is a
startup error, not a warning that scrolls past.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "ci", "staging", "production"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ALETHEIA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = "local"
    database_url: str | None = None
    secret_key: SecretStr | None = None
    artifact_root: Path = Path("artifacts")
    max_artifact_bytes: int = Field(default=256 * 1024 * 1024, gt=0)
    worker_processes: int | None = Field(default=None, ge=1, le=32)
    explainer_max_rows: int = Field(default=5_000, gt=0)
    log_level: str = "INFO"
    log_json: bool = False

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @model_validator(mode="after")
    def _production_requires_secrets(self) -> Settings:
        if self.environment != "production":
            return self
        missing = []
        if not self.database_url:
            missing.append("ALETHEIA_DATABASE_URL")
        if self.secret_key is None or not self.secret_key.get_secret_value():
            missing.append("ALETHEIA_SECRET_KEY")
        if missing:
            raise ValueError(f"production requires {', '.join(missing)}")
        return self

    def resolved_worker_processes(self) -> int:
        """Worker count for the explanation pool: explicit, else CPU-bound with a small ceiling.

        The ceiling is deliberate. Explanation work is memory-heavy per process and the pool exists to
        escape the GIL, not to fill every core (see docs/03-system-architecture.md §5.1).
        """
        if self.worker_processes is not None:
            return self.worker_processes
        return max(1, min(4, (os.cpu_count() or 2) - 1))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
