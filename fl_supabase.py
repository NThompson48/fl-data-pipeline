"""
Shared Supabase connection helper for FLHitting notebooks/scripts.

Reads credentials from secrets.toml (same directory as this file) instead of
hardcoding them inline. Keep secrets.toml out of version control.

Usage:
    from fl_supabase import get_engine
    import pandas as pd
    from sqlalchemy import text

    q = text("SELECT * FROM pitches WHERE batter_team = :team")
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn, params={"team": "Mississippi Mud Monsters"})
"""

import tomllib
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

SECRETS_PATH = Path(__file__).resolve().parent / "secrets.toml"


def _load_secrets() -> dict:
    if not SECRETS_PATH.exists():
        raise FileNotFoundError(
            f"secrets.toml not found at {SECRETS_PATH}. "
            "Create it with a [supabase] section (host, port, database, user, password)."
        )
    with open(SECRETS_PATH, "rb") as f:
        return tomllib.load(f)


def get_engine(statement_timeout_ms: int = 300_000):
    """Return a SQLAlchemy engine connected to the Supabase Postgres pooler.

    statement_timeout_ms raises the per-session query timeout (default 5 min)
    since the pooler's default is too short for large pulls.
    """
    cfg = _load_secrets()["supabase"]
    url = (
        f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={
            "sslmode": "require",
            "options": f"-c statement_timeout={statement_timeout_ms}",
        },
    )


def read_sql_df(query, params: dict | None = None, statement_timeout_ms: int = 300_000) -> pd.DataFrame:
    """Run a query and return a DataFrame, with statement_timeout guaranteed to
    apply (the pooler doesn't reliably honor connect_args options, so it's set
    explicitly inside the same transaction/backend connection here).
    """
    engine = get_engine(statement_timeout_ms=statement_timeout_ms)
    with engine.begin() as conn:
        conn.execute(text(f"SET statement_timeout = {statement_timeout_ms}"))
        return pd.read_sql(query, conn, params=params)
