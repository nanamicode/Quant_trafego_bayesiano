from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .reproducibility import dataframe_fingerprint


class LocalWarehouse:
    """Embedded local analytical store with no server process."""

    def __init__(self, root: str | Path = "workspace"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.root / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "quant_trafego.duckdb"
        self._init_schema()

    def connect(self):
        return duckdb.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self.connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    data_sha256 VARCHAR PRIMARY KEY,
                    parquet_path VARCHAR NOT NULL,
                    rows BIGINT NOT NULL,
                    created_at TIMESTAMP DEFAULT current_timestamp
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id VARCHAR PRIMARY KEY,
                    data_sha256 VARCHAR NOT NULL,
                    inference_mode VARCHAR,
                    created_at TIMESTAMP,
                    manifest_json VARCHAR NOT NULL
                )
                """
            )

    def store_snapshot(
        self,
        df: pd.DataFrame,
        data_hash: str | None = None,
    ) -> tuple[str, Path]:
        digest = data_hash or dataframe_fingerprint(df)
        path = self.snapshots_dir / f"{digest}.parquet"

        if not path.exists():
            with self.connect() as con:
                relation = con.from_df(df)
                relation.write_parquet(str(path), compression="zstd")

        with self.connect() as con:
            con.execute(
                """
                INSERT OR IGNORE INTO snapshots(data_sha256, parquet_path, rows)
                VALUES (?, ?, ?)
                """,
                [digest, str(path.resolve()), int(len(df))],
            )
        return digest, path

    def register_run(self, manifest: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO runs(
                    run_id, data_sha256, inference_mode, created_at, manifest_json
                )
                VALUES (?, ?, ?, CAST(? AS TIMESTAMP), ?)
                """,
                [
                    manifest["run_id"],
                    manifest["data_sha256"],
                    manifest.get("inference_mode"),
                    manifest["created_at_utc"],
                    json.dumps(manifest, ensure_ascii=False, default=str),
                ],
            )

    def list_runs(self) -> pd.DataFrame:
        with self.connect() as con:
            return con.execute(
                """
                SELECT run_id, data_sha256, inference_mode, created_at
                FROM runs
                ORDER BY created_at DESC
                """
            ).df()