from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd

from . import __version__
from .hardware import detect_hardware


_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "duckdb",
    "pymc",
    "arviz",
    "pymc-marketing",
    "numpyro",
    "botorch",
    "ax-platform",
    "mabwiser",
)


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    canonical = df.copy()
    canonical.columns = [str(c) for c in canonical.columns]

    if "date" in canonical.columns:
        canonical["date"] = pd.to_datetime(
            canonical["date"], errors="coerce"
        ).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

    columns = sorted(canonical.columns)
    canonical = canonical[columns]

    sort_keys = [
        c
        for c in ["date", "campaign_id", "adset_id", "ad_id"]
        if c in canonical.columns
    ]
    if sort_keys:
        canonical = canonical.sort_values(sort_keys, kind="mergesort")
    canonical = canonical.reset_index(drop=True)

    payload = canonical.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        value = result.stdout.strip()
        return value or None
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _serialize_config(config: Any) -> Any:
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json")
    if isinstance(config, dict):
        return config
    return str(config)


def build_run_manifest(
    df: pd.DataFrame,
    *,
    config: Any,
    inference_mode: str,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data_hash = dataframe_fingerprint(df)
    hardware = detect_hardware()
    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{data_hash[:10]}"

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "data_sha256": data_hash,
        "rows": int(len(df)),
        "date_min": (
            str(pd.to_datetime(df["date"]).min())
            if "date" in df.columns and len(df)
            else None
        ),
        "date_max": (
            str(pd.to_datetime(df["date"]).max())
            if "date" in df.columns and len(df)
            else None
        ),
        "inference_mode": inference_mode,
        "seed": int(seed),
        "config": _serialize_config(config),
        "package_version": __version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "hardware": asdict(hardware),
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_run_manifest(
    manifest: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path