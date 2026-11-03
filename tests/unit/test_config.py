"""Configuration must fail loudly in production and default safely everywhere else."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aletheia.core.config import Settings
from aletheia.core.logging import configure_logging


def test_defaults_are_local_and_safe() -> None:
    settings = Settings(_env_file=None)
    assert settings.environment == "local"
    assert settings.is_production is False
    assert settings.database_url is None
    assert settings.secret_key is None
    assert 1 <= settings.resolved_worker_processes() <= 4


def test_production_requires_database_and_secret() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, environment="production")
    message = str(excinfo.value)
    assert "ALETHEIA_DATABASE_URL" in message
    assert "ALETHEIA_SECRET_KEY" in message


def test_production_requires_secret_even_when_database_is_set() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Settings(
            _env_file=None,
            environment="production",
            database_url="postgresql+psycopg://user:pass@localhost:5432/aletheia",
        )
    assert "ALETHEIA_SECRET_KEY" in str(excinfo.value)


def test_production_accepts_a_complete_configuration() -> None:
    settings = Settings(
        _env_file=None,
        environment="production",
        database_url="postgresql+psycopg://user:pass@localhost:5432/aletheia",
        secret_key="not-a-real-secret",
    )
    assert settings.is_production is True
    assert settings.secret_key is not None
    assert "not-a-real-secret" not in repr(settings)


def test_worker_count_can_be_pinned() -> None:
    assert Settings(_env_file=None, worker_processes=2).resolved_worker_processes() == 2


def test_unknown_log_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging("LOUD")


def test_json_logging_is_configurable() -> None:
    configure_logging("INFO", json=True)  # must not raise
    configure_logging("DEBUG", json=False)
