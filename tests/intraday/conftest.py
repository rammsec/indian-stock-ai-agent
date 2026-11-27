"""Shared pytest fixtures for intraday test isolation.

Redirects 5-minute Parquet cache writes (`trader.data.loader.CACHE_DIR` and
`trader.lab.data_service.CACHE_DIR`) into a per-session temporary directory
so running `pytest tests/intraday/` never leaves untracked
`data/intraday/*_NSE_5m.parquet` files in the repository working tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import trader.data.loader as _data_loader
import trader.lab.data_service as _lab_data_service


@pytest.fixture(autouse=True)
def _isolate_intraday_cache_dir(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    cache_dir = tmp_path_factory.mktemp("intraday_cache")
    monkeypatch.setattr(_data_loader, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(_lab_data_service, "CACHE_DIR", cache_dir)
    return cache_dir
