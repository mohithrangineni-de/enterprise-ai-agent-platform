"""
tools/database.py

Async PostgreSQL wrapper using SQLAlchemy + asyncpg.
Used by the SQL Agent to execute generated queries.
"""

from __future__ import annotations
from typing import Any

from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection
from sqlalchemy import text

from core.config import settings
from observability.logger import get_logger

log = get_logger(__name__)

# Pre-defined schema description injected into the SQL agent prompt.
# In production: auto-generate this by introspecting the DB at startup.
SCHEMA_DESCRIPTION = """
Tables:
- sales(id, date, region, product_id, revenue, units_sold, channel)
- products(id, name, category, price, launch_date)
- customers(id, region, segment, acquisition_date)
- targets(period, region, target_revenue)

Key relationships:
- sales.product_id → products.id
- sales.region matches customers.region, targets.region

Useful columns:
- Use sales.date for time-based filtering (YYYY-MM-DD format)
- Quarters: Q1=Jan–Mar, Q2=Apr–Jun, Q3=Jul–Sep, Q4=Oct–Dec
- Regions: APAC, EMEA, AMER, LATAM
"""


class DatabaseTool:
    def __init__(self):
        self._engine = create_async_engine(
            settings.db_url,
            pool_size=5,
            echo=False,
        )

    async def get_schema_description(self) -> str:
        return SCHEMA_DESCRIPTION

    async def execute(self, sql: str) -> tuple[list[dict], list[str]]:
        """
        Execute a SELECT query and return (rows, column_names).
        Rows are returned as dicts for easy JSON serialization.
        """
        log.info("db.execute", sql_preview=sql[:80])
        async with self._engine.connect() as conn:
            result = await conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
            log.info("db.execute.done", rows=len(rows), columns=columns)
            return rows, columns

    async def close(self):
        await self._engine.dispose()
