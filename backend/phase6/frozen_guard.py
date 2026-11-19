"""Guard proving frozen assets are untouched. Read-only: only hashes files."""
from __future__ import annotations

import glob
import hashlib
from pathlib import Path
from typing import Iterable, Optional

_N = "backend/news_intel/"
FROZEN_PATHS: list[str] = [
    "backend/prediction/*.py",
    "backend/features/*.py",
    *[f"{_N}{n}.py" for n in ("features", "combiner", "forward_validation", "pipeline",
                              "extractor", "collector", "cross_stock", "monitoring")],
    "data/predictions/*.parquet", "data/predictions/*.json",
    "data/processed/*.parquet", "data/features/*", "data/backtests/*",
    ".github/workflows/*.yml", "requirements.txt",
]


class FrozenFileChanged(AssertionError):
    pass


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:  # read-only
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_frozen_files(paths: Optional[Iterable[str]] = None,
                          root: str | Path = ".") -> dict[str, str]:
    """Map relative file path -> sha256 for every file matching the globs."""
    root = Path(root)
    snap: dict[str, str] = {}
    for pat in (FROZEN_PATHS if paths is None else paths):
        for m in sorted(glob.glob(str(root / pat))):
            p = Path(m)
            if p.is_file():
                snap[str(p.relative_to(root))] = _sha(p)
    return snap


def assert_unchanged(snapshot: dict[str, str], root: str | Path = ".") -> None:
    """Raise :class:`FrozenFileChanged` if any snapshotted file was modified or deleted."""
    root = Path(root)
    problems = []
    for rel, digest in snapshot.items():
        p = root / rel
        if not p.is_file():
            problems.append(f"deleted: {rel}")
        elif _sha(p) != digest:
            problems.append(f"modified: {rel}")
    if problems:
        raise FrozenFileChanged("; ".join(problems))
