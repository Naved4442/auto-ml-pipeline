from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("Missing required environment variable: DATABASE_URL")
    return database_url


def get_engine() -> Engine:
    return create_engine(get_database_url())
