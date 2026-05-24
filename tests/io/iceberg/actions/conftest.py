"""Fixtures for Iceberg action tests (rewrite_data_files etc.).

Uses pyiceberg's SqlCatalog on a temp dir — no docker required.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pyiceberg = pytest.importorskip("pyiceberg")

from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField, StringType


@pytest.fixture(scope="function")
def local_catalog(tmp_path):
    catalog = SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path}/pyiceberg_catalog.db",
        warehouse=f"file://{tmp_path}",
    )
    catalog.create_namespace("default")
    yield catalog
    catalog.engine.dispose()


@pytest.fixture
def simple_schema() -> Schema:
    return Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "label", StringType(), required=False),
    )


def _tiny_arrow_table(start: int, n: int) -> pa.Table:
    return pa.table(
        {
            "id": pa.array(list(range(start, start + n)), type=pa.int64()),
            "label": pa.array(
                [f"row-{i}" for i in range(start, start + n)], type=pa.string()
            ),
        }
    )


@pytest.fixture
def make_tiny_table(local_catalog, simple_schema):
    """Create an unpartitioned table seeded with `n_files` tiny appends."""

    def _make(name: str = "default.tiny", n_files: int = 12, rows_per_file: int = 4):
        table = local_catalog.create_table(
            identifier=name,
            schema=simple_schema,
            partition_spec=UNPARTITIONED_PARTITION_SPEC,
        )
        for i in range(n_files):
            table.append(_tiny_arrow_table(i * rows_per_file, rows_per_file))
        return table

    return _make
