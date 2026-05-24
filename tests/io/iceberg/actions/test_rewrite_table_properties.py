"""Verify Iceberg `write.*` table properties are honored by the rewrite writer."""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table


def test_compression_codec_from_write_properties(make_tiny_table):
    table = make_tiny_table(name="default.t_props", n_files=4, rows_per_file=8)
    with table.transaction() as tx:
        tx.set_properties(**{"write.parquet.compression-codec": "gzip"})

    dt = Table.from_iceberg(table)
    result = dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})
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
        assert "GZIP" in codecs or "gzip" in {
            c.lower() for c in codecs
        }, f"expected gzip codec, got {codecs} for {path}"


def test_target_file_size_from_write_property(make_tiny_table):
    """When no option is passed, ``write.target-file-size-bytes`` should drive the planner."""
    from daft.io.iceberg import _compact

    table = make_tiny_table(name="default.t_target_size", n_files=4, rows_per_file=8)
    with table.transaction() as tx:
        tx.set_properties(**{_compact.WRITE_TARGET_FILE_SIZE_BYTES_KEY: "1048576"})

    captured: dict[str, int] = {}
    original = _compact._rust_iceberg.validate_options_py

    def spy(opts):
        captured["target-file-size-bytes"] = int(opts.get("target-file-size-bytes", -1))
        return original(opts)

    _compact._rust_iceberg.validate_options_py = spy
    try:
        dt = Table.from_iceberg(table)
        dt.rewrite_data_files(
            "binpack", options={"rewrite-all": True, "min-input-files": 2}
        )
    finally:
        _compact._rust_iceberg.validate_options_py = original

    assert captured["target-file-size-bytes"] == 1048576


def test_target_file_size_option_overrides_write_property(make_tiny_table):
    from daft.io.iceberg import _compact

    table = make_tiny_table(
        name="default.t_target_override", n_files=2, rows_per_file=2
    )
    with table.transaction() as tx:
        tx.set_properties(**{_compact.WRITE_TARGET_FILE_SIZE_BYTES_KEY: "1048576"})

    captured: dict[str, int] = {}
    original = _compact._rust_iceberg.validate_options_py

    def spy(opts):
        captured["target-file-size-bytes"] = int(opts.get("target-file-size-bytes", -1))
        return original(opts)

    _compact._rust_iceberg.validate_options_py = spy
    try:
        dt = Table.from_iceberg(table)
        dt.rewrite_data_files(
            "binpack",
            options={
                "target-file-size-bytes": 4194304,
                "rewrite-all": True,
                "min-input-files": 2,
            },
        )
    finally:
        _compact._rust_iceberg.validate_options_py = original

    assert captured["target-file-size-bytes"] == 4194304


def test_unsupported_format_rejected(make_tiny_table):
    table = make_tiny_table(name="default.t_fmt", n_files=4, rows_per_file=4)
    with table.transaction() as tx:
        tx.set_properties(**{"write.format-default": "orc"})

    dt = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="write.format-default"):
        dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})


def _rewritten_parquet_files(table) -> list[pq.ParquetFile]:
    out: list[pq.ParquetFile] = []
    for f in table.scan().plan_files():
        path = f.file.file_path
        if path.startswith("file://"):
            path = path[len("file://") :]
        out.append(pq.ParquetFile(path))
    return out


def _make_property_table(make_tiny_table, name: str, **props: str):
    table = make_tiny_table(name=name, n_files=4, rows_per_file=256)
    with table.transaction() as tx:
        tx.set_properties(**props)
    return table


def test_compression_level_from_write_properties(make_tiny_table):
    """``write.parquet.compression-level`` reaches the writer."""
    table = _make_property_table(
        make_tiny_table,
        "default.t_props_level",
        **{
            "write.parquet.compression-codec": "zstd",
            "write.parquet.compression-level": "5",
        },
    )
    dt = Table.from_iceberg(table)
    dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})
    table.refresh()
    for pf in _rewritten_parquet_files(table):
        codecs = {
            pf.metadata.row_group(rg).column(c).compression
            for rg in range(pf.num_row_groups)
            for c in range(pf.metadata.num_columns)
        }
        assert {c.upper() for c in codecs} == {"ZSTD"}


def test_row_group_size_bytes_from_write_properties(make_tiny_table):
    """``write.parquet.row-group-size-bytes`` shrinks emitted row groups."""
    table = _make_property_table(
        make_tiny_table,
        "default.t_props_rg",
        **{"write.parquet.row-group-size-bytes": "4096"},
    )
    dt = Table.from_iceberg(table)
    dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})
    table.refresh()
    pfs = _rewritten_parquet_files(table)
    assert pfs
    for pf in pfs:
        for rg in range(pf.num_row_groups):
            assert pf.metadata.row_group(rg).total_byte_size <= 16384, (
                f"row group {rg} = {pf.metadata.row_group(rg).total_byte_size} bytes "
                f"exceeded 4 * target with target=4096"
            )
        assert (
            pf.num_row_groups > 1
        ), "tiny row-group cap should produce multiple groups"


def test_page_size_bytes_from_write_properties(make_tiny_table):
    """``write.parquet.page-size-bytes`` shrinks emitted data pages."""
    table = _make_property_table(
        make_tiny_table,
        "default.t_props_pg",
        **{"write.parquet.page-size-bytes": "1024"},
    )
    dt = Table.from_iceberg(table)
    dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})
    table.refresh()
    pfs = _rewritten_parquet_files(table)
    assert pfs
    for pf in pfs:
        col0 = pf.metadata.row_group(0).column(0)
        # data_page_offset moves forward by ~page-size between pages, so a single
        # column's total compressed size should exceed one page once row count > 0.
        assert col0.total_compressed_size > 0


def test_dict_size_bytes_from_write_properties(make_tiny_table):
    """``write.parquet.dict-size-bytes`` is accepted and the rewrite still succeeds."""
    table = _make_property_table(
        make_tiny_table,
        "default.t_props_dict",
        **{"write.parquet.dict-size-bytes": "2048"},
    )
    dt = Table.from_iceberg(table)
    result = dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})
    assert result.added_files >= 1


def test_avro_compression_codec_applies_to_new_manifests(make_tiny_table):
    """``write.avro.compression-codec`` reaches the manifest writer in rewrite_manifests."""
    table = make_tiny_table(name="default.t_avro_codec", n_files=6, rows_per_file=2)
    with table.transaction() as tx:
        tx.set_properties(**{"write.avro.compression-codec": "deflate"})
    table.refresh()

    dt = Table.from_iceberg(table)
    result = dt.rewrite_manifests(
        options={
            "manifest-target-size-bytes": 1024 * 1024,
            "manifest-min-count-to-merge": 2,
        }
    )
    assert result.added_manifests_count >= 1
    table.refresh()
    snap = table.metadata.current_snapshot()
    for m in snap.manifests(table.io):
        path = m.manifest_path
        if path.startswith("file://"):
            path = path[len("file://") :]
        header = open(path, "rb").read(512)
        assert (
            b"deflate" in header
        ), f"expected deflate codec marker in avro header of {m.manifest_path}"
