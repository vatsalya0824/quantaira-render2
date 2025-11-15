# backend/db.py
import os
import sqlalchemy as sa
from sqlalchemy import text

DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = sa.create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)

def init_db():
    schema = """
    CREATE TABLE IF NOT EXISTS measurements (
      id SERIAL PRIMARY KEY,
      created_utc TIMESTAMPTZ NOT NULL,
      metric TEXT NOT NULL,
      value_1 DOUBLE PRECISION,
      value_2 DOUBLE PRECISION,
      device_id TEXT,
      device_name TEXT,
      raw JSONB
    );

    CREATE INDEX IF NOT EXISTS idx_measurements_created ON measurements (created_utc);
    CREATE INDEX IF NOT EXISTS idx_measurements_metric  ON measurements (metric);
    """
    with engine.begin() as conn:
        conn.execute(text(schema))
