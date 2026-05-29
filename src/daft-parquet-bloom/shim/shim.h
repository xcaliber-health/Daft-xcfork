#ifndef DAFT_PARQUET_BLOOM_SHIM_H
#define DAFT_PARQUET_BLOOM_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle owned by the C++ side. Created by daft_bloom_open_*, freed
 * by daft_bloom_close. NOT thread-safe; each handle is owned by one caller. */
typedef struct DaftBloomReader DaftBloomReader;

/* Open a local-filesystem parquet file by path. Returns NULL on error; call
 * daft_bloom_last_error() for a description.
 *
 * Use this only for plain file:// access. For S3/GCS/ABFS use
 * daft_bloom_open_from_pyarrow_native_file(): pyiceberg's FileIO yields a
 * pyarrow.NativeFile that already speaks every protocol pyarrow knows. */
DaftBloomReader* daft_bloom_open_local(const char* path);

/* Open from a pyarrow.NativeFile passed as a PyObject*. The shim wraps it
 * via arrow::py::PyReadableFile (borrowed PyObject*, no INCREF) and
 * constructs a ParquetFileReader on top. Returns NULL on error.
 *
 * IMPORTANT LIFETIME CONTRACT: PyReadableFile holds a borrowed reference
 * to py_native_file. The caller MUST keep py_native_file alive (i.e.
 * INCREF'd at the Python level) for the entire lifetime of the returned
 * DaftBloomReader, NOT just for the duration of this call. Dropping the
 * Python NativeFile before daft_bloom_close() will crash the next probe.
 *
 * In daft's Python orchestrator this is enforced by holding the
 * NativeFile reference in a try/finally that outlives the reader. */
DaftBloomReader* daft_bloom_open_from_pyarrow_native_file(void* py_native_file);

void daft_bloom_close(DaftBloomReader* reader);

/* Number of row groups in the file. Returns -1 on error. */
int32_t daft_bloom_num_row_groups(const DaftBloomReader* reader);

/* Probe a single (row group, column) for a literal.
 *
 * column_path: dot-separated leaf path (top-level columns are just "name").
 * type_id:     parquet::Type::type enum value:
 *                BOOLEAN=0, INT32=1, INT64=2, INT96=3, FLOAT=4, DOUBLE=5,
 *                BYTE_ARRAY=6, FIXED_LEN_BYTE_ARRAY=7.
 *              INT8/INT16/UINT* must be widened to INT32 by the caller.
 *              UINT64 reinterprets as INT64.
 *              UUID is FIXED_LEN_BYTE_ARRAY(16).
 * value_bytes: raw bytes whose layout matches the parquet physical type.
 *
 * Returns:
 *    1  bloom says "probably present" — caller must keep this row group
 *    0  bloom says "definitely absent" — caller may prune this row group
 *   -1  no bloom filter for this column in this row group (inconclusive)
 *   -2  error; daft_bloom_last_error() has details */
int32_t daft_bloom_probe(
    DaftBloomReader* reader,
    int32_t row_group,
    const char* column_path,
    int32_t type_id,
    const uint8_t* value_bytes,
    size_t value_len);

/* Thread-local last-error string. Lifetime: valid until the next call into
 * this shim on the same thread. Returns NULL if no error is set. */
const char* daft_bloom_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* DAFT_PARQUET_BLOOM_SHIM_H */
