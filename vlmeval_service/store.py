from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
import sqlite3
from threading import RLock


JSON_FIELDS = {"config", "execution_status", "result"}


class TaskStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    quality_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    parent_task_id TEXT,
                    idempotency_key TEXT UNIQUE,
                    config TEXT NOT NULL,
                    execution_status TEXT NOT NULL,
                    result TEXT NOT NULL,
                    error TEXT,
                    work_dir TEXT NOT NULL,
                    pid INTEGER
                )
                """
            )
            connection.commit()

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        task = dict(row)
        for field in JSON_FIELDS:
            task[field] = json.loads(task[field])
        return task

    def create(self, task: dict) -> None:
        fields = list(task)
        values = [json.dumps(task[f], ensure_ascii=False) if f in JSON_FIELDS else task[f] for f in fields]
        placeholders = ",".join("?" for _ in fields)
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                f"INSERT INTO tasks ({','.join(fields)}) VALUES ({placeholders})",
                values,
            )
            connection.commit()

    def update(self, task_id: str, **changes) -> None:
        if not changes:
            return
        fields = list(changes)
        values = [json.dumps(changes[f], ensure_ascii=False) if f in JSON_FIELDS else changes[f] for f in fields]
        values.append(task_id)
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                f"UPDATE tasks SET {','.join(f'{field}=?' for field in fields)} WHERE task_id=?",
                values,
            )
            connection.commit()

    def get(self, task_id: str) -> dict | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return self._decode(row)

    def by_idempotency_key(self, key: str | None) -> dict | None:
        if not key:
            return None
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE idempotency_key=?", (key,)
            ).fetchone()
        return self._decode(row)

    def list(self, limit: int = 50) -> list[dict]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode(row) for row in rows]

    def active(self) -> list[dict]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE status IN ('queued','running','pausing')"
            ).fetchall()
        return [self._decode(row) for row in rows]

    def mark_interrupted(self, updated_at: str) -> None:
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE tasks
                SET status='interrupted', quality_status='partial', updated_at=?,
                    error='service restarted before task completion', pid=NULL
                WHERE status IN ('queued','running','pausing')
                """,
                (updated_at,),
            )
            connection.commit()
