"""Readiness probes.

Liveness answers "is this process alive". Readiness answers "can this process do its job", which for this
service means reaching its database. A probe never raises and never echoes a connection string: a health
endpoint that leaks a password into a log aggregator is a vulnerability, not an observability feature.
"""

from __future__ import annotations


def check_database(url: str | None, *, connect_timeout: int = 2) -> tuple[bool, str]:
    """Return ``(ok, detail)`` for the configured database.

    ``detail`` is a short, non-sensitive status: a driver name, a missing-configuration marker, or an
    exception class. It never contains the URL, the user, or the password.
    """
    if not url:
        return False, "not_configured"

    try:
        from sqlalchemy import create_engine, text
    except ModuleNotFoundError:  # pragma: no cover - dependency is declared
        return False, "driver_unavailable"

    connect_args: dict[str, object] = {}
    if url.startswith("postgresql"):
        connect_args["connect_timeout"] = connect_timeout

    engine = None
    try:
        engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True, "ok"
    except Exception as exc:
        return False, type(exc).__name__
    finally:
        if engine is not None:
            engine.dispose()
