// C++ shim that wraps libparquet's bloom-filter read API.
//
// Why this exists: PyArrow ships libparquet inside its wheel with full bloom
// filter support, but the Cython wrapper in `_parquet.pyx` does not expose
// the BloomFilter / BloomFilterReader classes. This shim links directly
// against libparquet's C++ ABI and re-exports a stable C interface that
// Rust/PyO3 can call.
//
// ABI risk: libparquet is a C++ library; symbol names change across Arrow
// versions. We pin compatibility by linking against the libparquet that
// shipped with the pyarrow build-time interpreter (see build.rs). At
// runtime the same pyarrow must be importable before this shim's symbols
// resolve — the Python entrypoint preloads pyarrow's libs.

#include "shim.h"

// Python.h MUST come before any standard headers per CPython convention,
// otherwise the _POSIX_C_SOURCE defines clash on some platforms.
#include <Python.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <arrow/io/interfaces.h>
#include <arrow/python/io.h>
#include <arrow/python/pyarrow.h>
#include <parquet/bloom_filter.h>
#include <parquet/bloom_filter_reader.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>
#include <parquet/properties.h>
#include <parquet/types.h>

namespace {

// Thread-local error buffer. The C API returns a `const char*` that the
// caller must use before issuing another call from the same thread.
thread_local std::string g_last_error;

void set_error(const std::string& msg) { g_last_error = msg; }
void clear_error() { g_last_error.clear(); }

// arrow::py::import_pyarrow() must be invoked once per process before any
// arrow_python helper is called. We guard with std::once_flag; the helper
// itself is allowed to be called many times but doing so is wasteful.
std::once_flag g_pyarrow_init;
int g_pyarrow_init_rc = 0;  // 0 on success; nonzero is the failure code.

bool ensure_pyarrow_initialized() {
  std::call_once(g_pyarrow_init, []() {
    g_pyarrow_init_rc = arrow::py::import_pyarrow();
  });
  if (g_pyarrow_init_rc != 0) {
    set_error("arrow::py::import_pyarrow() failed; check that pyarrow is "
              "importable from the same Python interpreter that loaded daft");
    return false;
  }
  return true;
}

// Build a path->leaf-index map once per file. Used by daft_bloom_probe so
// repeated probes against the same file (the common case during scan
// planning) are O(1) instead of O(num_columns) per call.
//
// Iceberg/Parquet bloom is leaf-only on flat columns; the dotted-path form
// leaves room for nested columns without a schema change.
std::unordered_map<std::string, int> build_column_index_map(
    const parquet::SchemaDescriptor* schema) {
  std::unordered_map<std::string, int> out;
  const int n = schema->num_columns();
  out.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    out.emplace(schema->Column(i)->path()->ToDotString(), i);
  }
  return out;
}

// Dispatch a `Hash(value)` call to the bloom filter, picking the c_type
// associated with the requested parquet physical type. Returns the 64-bit
// hash; the caller checks via FindHash().
//
// Variable-length types (BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY) take separate
// paths in daft_bloom_probe() because their Hash() overloads accept a
// pointer + (for FLBA) a length.
uint64_t hash_fixed_width(parquet::BloomFilter* bf,
                          parquet::Type::type t,
                          const uint8_t* bytes,
                          size_t len) {
  switch (t) {
    case parquet::Type::INT32: {
      if (len != sizeof(int32_t)) {
        set_error("INT32 literal must be exactly 4 bytes");
        return 0;
      }
      int32_t v;
      std::memcpy(&v, bytes, sizeof(v));
      return bf->Hash(v);
    }
    case parquet::Type::INT64: {
      if (len != sizeof(int64_t)) {
        set_error("INT64 literal must be exactly 8 bytes");
        return 0;
      }
      int64_t v;
      std::memcpy(&v, bytes, sizeof(v));
      return bf->Hash(v);
    }
    case parquet::Type::FLOAT: {
      if (len != sizeof(float)) {
        set_error("FLOAT literal must be exactly 4 bytes");
        return 0;
      }
      float v;
      std::memcpy(&v, bytes, sizeof(v));
      return bf->Hash(v);
    }
    case parquet::Type::DOUBLE: {
      if (len != sizeof(double)) {
        set_error("DOUBLE literal must be exactly 8 bytes");
        return 0;
      }
      double v;
      std::memcpy(&v, bytes, sizeof(v));
      return bf->Hash(v);
    }
    default:
      set_error("type_id not supported by hash_fixed_width");
      return 0;
  }
}

}  // namespace

struct DaftBloomReader {
  std::unique_ptr<parquet::ParquetFileReader> file_reader;
  // `BloomFilterReader&` is borrowed from `file_reader`. Stored as a pointer
  // so we can construct DaftBloomReader after the file_reader is moved in.
  parquet::BloomFilterReader* bloom_reader = nullptr;
  const parquet::SchemaDescriptor* schema = nullptr;
  // Cached path->leaf-index lookup. Built once at open, reused per probe.
  std::unordered_map<std::string, int> column_index_by_path;
};

extern "C" {

DaftBloomReader* daft_bloom_open_local(const char* path) {
  clear_error();
  try {
    auto reader = parquet::ParquetFileReader::OpenFile(
        path, /*memory_map=*/false, parquet::default_reader_properties());
    if (!reader) {
      set_error("ParquetFileReader::OpenFile returned null");
      return nullptr;
    }

    auto out = std::make_unique<DaftBloomReader>();
    out->bloom_reader = &reader->GetBloomFilterReader();
    out->schema = reader->metadata()->schema();
    out->column_index_by_path = build_column_index_map(out->schema);
    out->file_reader = std::move(reader);
    return out.release();
  } catch (const std::exception& e) {
    set_error(std::string("OpenFile threw: ") + e.what());
    return nullptr;
  } catch (...) {
    set_error("OpenFile threw unknown exception");
    return nullptr;
  }
}

DaftBloomReader* daft_bloom_open_from_pyarrow_native_file(void* py_native_file) {
  clear_error();
  if (!py_native_file) {
    set_error("null PyObject*");
    return nullptr;
  }
  if (!ensure_pyarrow_initialized()) return nullptr;
  try {
    // arrow::py::unwrap_random_access_file was removed in Arrow 18 (PyArrow 23).
    // Use PyReadableFile which wraps any Python file-like object as a
    // RandomAccessFile — works for any pyarrow.NativeFile including S3-backed ones.
    auto* py_obj = static_cast<PyObject*>(py_native_file);
    std::shared_ptr<arrow::io::RandomAccessFile> arrow_file =
        std::make_shared<arrow::py::PyReadableFile>(py_obj);

    auto reader = parquet::ParquetFileReader::Open(
        arrow_file, parquet::default_reader_properties());
    if (!reader) {
      set_error("ParquetFileReader::Open returned null");
      return nullptr;
    }

    auto out = std::make_unique<DaftBloomReader>();
    out->bloom_reader = &reader->GetBloomFilterReader();
    out->schema = reader->metadata()->schema();
    out->column_index_by_path = build_column_index_map(out->schema);
    out->file_reader = std::move(reader);
    return out.release();
  } catch (const std::exception& e) {
    set_error(std::string("open_from_pyarrow_native_file threw: ") + e.what());
    return nullptr;
  } catch (...) {
    set_error("open_from_pyarrow_native_file threw unknown exception");
    return nullptr;
  }
}

void daft_bloom_close(DaftBloomReader* reader) {
  clear_error();
  delete reader;
}

int32_t daft_bloom_num_row_groups(const DaftBloomReader* reader) {
  if (!reader || !reader->file_reader) return -1;
  return reader->file_reader->metadata()->num_row_groups();
}

int32_t daft_bloom_probe(DaftBloomReader* reader,
                         int32_t row_group,
                         const char* column_path,
                         int32_t type_id,
                         const uint8_t* value_bytes,
                         size_t value_len) {
  if (!reader || !reader->bloom_reader) {
    set_error("null reader");
    return -2;
  }
  clear_error();
  try {
    auto it = reader->column_index_by_path.find(column_path);
    if (it == reader->column_index_by_path.end()) {
      set_error(std::string("column not in schema: ") + column_path);
      return -2;
    }
    const int col = it->second;

    auto rg_bloom_reader = reader->bloom_reader->RowGroup(row_group);
    if (!rg_bloom_reader) return -1;

    std::unique_ptr<parquet::BloomFilter> bf =
        rg_bloom_reader->GetColumnBloomFilter(col);
    if (!bf) return -1;

    if (value_len > static_cast<size_t>(UINT32_MAX)) {
      set_error("value_len exceeds uint32_t max");
      return -2;
    }
    uint64_t hash;
    const auto t = static_cast<parquet::Type::type>(type_id);
    if (t == parquet::Type::BYTE_ARRAY) {
      parquet::ByteArray ba(static_cast<uint32_t>(value_len), value_bytes);
      hash = bf->Hash(&ba);
    } else if (t == parquet::Type::FIXED_LEN_BYTE_ARRAY) {
      // FLBA hash length must match the column's declared type_length, not the
      // caller-supplied buffer length, or libparquet will hash the wrong byte
      // window and produce silent false negatives.
      const int32_t flba_len = reader->schema->Column(col)->type_length();
      if (static_cast<int32_t>(value_len) != flba_len) {
        set_error(std::string("FLBA value_len (") + std::to_string(value_len)
                  + ") does not match column type_length (" + std::to_string(flba_len) + ")");
        return -2;
      }
      parquet::FLBA flba(value_bytes);
      hash = bf->Hash(&flba, static_cast<uint32_t>(flba_len));
    } else {
      hash = hash_fixed_width(bf.get(), t, value_bytes, value_len);
      if (!g_last_error.empty()) return -2;
    }

    return bf->FindHash(hash) ? 1 : 0;
  } catch (const std::exception& e) {
    set_error(std::string("probe threw: ") + e.what());
    return -2;
  } catch (...) {
    set_error("probe threw unknown exception");
    return -2;
  }
}

const char* daft_bloom_last_error(void) {
  return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

}  // extern "C"
