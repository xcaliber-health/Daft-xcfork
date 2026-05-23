"""Verify Iceberg `write.*` table properties are honored by the rewrite writer."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table


def test_compression_codec_from_write_properties(make_tiny_table):
    table = make_tiny_table(name="default.t_props", n_files=4, rows_per_file=8)
    with table.transaction() as tx:
        tx.set_properties(**{"write.parquet.compression-codec": "gzip"})

    dt = Table.from_iceberg(table)
    result = dt.compact_files(
        options={"rewrite-all": True, "min-input-files": 2}
    )
    assert result.added_files >= 1
    table.refresh()

    # Inspect each rewritten parquet to confirm gzip compression.
    for f in table.scan().plan_files():
        path = f.file.file_path
        # Strip Iceberg-style scheme prefix for local FS reads.
        if path.startswith("file://"):
            path = path[len("file://") :]
        pf = pq.ParquetFile(path)
        codecs = {
            pf.metadata.row_group(rg).column(c).compression
            for rg in range(pf.num_row_groups)
            for c in range(pf.metadata.num_columns)
        }
        assert "GZIP" in codecs or "gzip" in {c.lower() for c in codecs}, (
            f"expected gzip codec, got {codecs} for {path}"
        )


def test_unsupported_format_rejected(make_tiny_table):
    table = make_tiny_table(name="default.t_fmt", n_files=4, rows_per_file=4)
    with table.transaction() as tx:
        tx.set_properties(**{"write.format-default": "orc"})

    dt = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="write.format-default"):
        dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})
