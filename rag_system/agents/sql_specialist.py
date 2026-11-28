"""
SQL Specialist Agent — answers questions by running safe SELECT queries
against a local SQLite database.

Safety rules (enforced, not LLM-promised):
- Only SELECT statements are executed.
- Queries that contain DDL/DML keywords are rejected before execution.
- Results are capped at 100 rows to avoid flooding the prompt.

The agent exposes a FastAPI service with a single tool: `sql_query`.
The Manager routes questions about structured / tabular data here.

Setup
-----
Point config.yaml → agents.sql_db_path at any SQLite file.
A sample DB is created automatically at data/sample.db if the path
does not exist (useful for demos and tests).
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, status
from loguru import logger

from rag_system.mcp_server.models import Tool, AgentRegistration, Message

# Keywords that must NOT appear in an allowed query
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|PRAGMA|ATTACH)\b",
    re.IGNORECASE,
)

MAX_ROWS = 100


def _is_safe(sql: str) -> bool:
    """Return True only if *sql* is a plain SELECT with no forbidden verbs."""
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        return False
    if _FORBIDDEN.search(stripped):
        return False
    return True


def _run_query(db_path: str, sql: str) -> Dict[str, Any]:
    """Execute a safe SELECT and return columns + rows."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(MAX_ROWS)
        columns = [d[0] for d in cur.description] if cur.description else []
        return {
            "columns": columns,
            "rows": [dict(r) for r in rows],
            "row_count": len(rows),
            "truncated": len(rows) == MAX_ROWS,
        }
    finally:
        conn.close()


def _get_schema(db_path: str) -> str:
    """Return a compact schema string (table + column names) for the prompt."""
    conn = sqlite3.connect(db_path)
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        lines = []
        for (tname,) in tables:
            cols = conn.execute(f"PRAGMA table_info({tname})").fetchall()
            col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            lines.append(f"  {tname}({col_defs})")
        return "Schema:\n" + "\n".join(lines) if lines else "No tables found."
    finally:
        conn.close()


def _ensure_sample_db(db_path: str) -> None:
    """Create a minimal demo database if *db_path* doesn't exist."""
    path = Path(db_path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT,
            auc_roc REAL,
            precision REAL,
            recall REAL,
            f1 REAL,
            created_at TEXT
        );
        INSERT OR IGNORE INTO models VALUES
            (1,'churn_v3','3.0',0.89,0.82,0.76,0.79,'2024-09-01'),
            (2,'revenue_forecast','1.2',0.91,0.85,0.80,0.82,'2024-09-15'),
            (3,'sentiment_classifier','2.1',0.87,0.78,0.81,0.79,'2024-10-01');

        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY,
            model_id INTEGER REFERENCES models(id),
            run_name TEXT,
            accuracy REAL,
            loss REAL,
            epochs INTEGER,
            notes TEXT
        );
        INSERT OR IGNORE INTO experiments VALUES
            (1,1,'baseline',0.84,0.42,50,'default hyperparams'),
            (2,1,'tuned_lr',0.87,0.38,80,'lr=0.001, dropout=0.3'),
            (3,2,'v1_run',0.88,0.35,100,'added feature engineering');
    """)
    conn.commit()
    conn.close()
    logger.info(f"Created sample SQLite DB at {db_path}")


class SQLSpecialistAgent:
    """
    Specialist that answers tabular / structured-data questions via SQLite.
    """

    def __init__(
        self,
        agent_id: str = "sql_specialist",
        name: str = "SQL Data Specialist",
        description: str = "Answers questions about structured tabular data using SQL",
        db_path: str = "data/sample.db",
        port: int = 8003,
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.db_path = db_path
        self.port = port

        _ensure_sample_db(db_path)
        self.schema = _get_schema(db_path)
        logger.info(f"SQLSpecialist ready — db={db_path}")

        self.app = FastAPI(title=f"Specialist: {name}", description=description)
        self._setup_routes()

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="sql_query",
                description=(
                    "Run a safe SELECT query against the SQLite database. "
                    f"Returns columns and rows as JSON.\n{self.schema}"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SQL SELECT statement"},
                    },
                    "required": ["query"],
                },
            )
        ]

    def get_registration(self, endpoint_host: str = "localhost") -> AgentRegistration:
        return AgentRegistration(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            tools=self.get_tools(),
            endpoint=f"http://{endpoint_host}:{self.port}",
        )

    def _setup_routes(self) -> None:

        @self.app.post("/execute", response_model=Dict)
        async def execute_tool(message: Message):
            if message.tool != "sql_query":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tool '{message.tool}' not supported",
                )
            sql = message.parameters.get("query", "").strip()
            if not sql:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing required parameter: query",
                )
            if not _is_safe(sql):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only SELECT queries are allowed.",
                )
            try:
                result = _run_query(self.db_path, sql)
                return {
                    "agent_id": self.agent_id,
                    "agent_name": self.name,
                    "query": sql,
                    "schema": self.schema,
                    **result,
                }
            except sqlite3.Error as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"SQL error: {exc}",
                )

        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "db_path": self.db_path,
                "schema": self.schema,
            }
