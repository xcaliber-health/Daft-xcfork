//! Build script: compile the C++ shim that wraps libparquet's bloom-filter API
//! and link it against the PyArrow-shipped libparquet/libarrow/libarrow_python.
//!
//! PyArrow ships its C++ libraries inside the wheel under `pyarrow/`. We
//! discover their location at build time by invoking the Python interpreter
//! used by maturin (set via the `PYO3_PYTHON` env var by pyo3-build-config,
//! or falling back to the `python` on PATH).
//!
//! At runtime the loaded `daft` extension will see undefined symbols that
//! resolve against PyArrow's libraries — these MUST be preloaded into the
//! process before `daft` is imported. See
//! `daft/_parquet_bloom/__init__.py::_preload_pyarrow_libs`.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Build-time paths discovered from the Python interpreter.
struct DiscoveredPaths {
    pyarrow_includes: Vec<String>,
    pyarrow_libdirs: Vec<String>,
    python_include: String,
    pyarrow_version: String,
}

/// Ask the build-time Python for the paths needed to compile against
/// PyArrow's bundled libparquet/libarrow/libarrow_python plus Python.h
/// (required because the shim consumes PyObject* via arrow::py helpers).
///
/// Output format from the embedded Python is line-oriented to avoid taking
/// a JSON dependency in build-deps:
///
/// ```text
/// VERSION <pyarrow version>
/// PY_INCLUDE <Python.h include dir>
/// PA_INCLUDE <pyarrow include dir>
/// LIBDIR <pyarrow lib dir, repeated per dir>
/// END
/// ```
fn discover_paths() -> DiscoveredPaths {
    let python = env::var("PYO3_PYTHON").unwrap_or_else(|_| "python".to_string());
    let script = r#"
import sys, sysconfig
try:
    import pyarrow
except Exception as e:
    sys.stderr.write(f"pyarrow not importable: {e}\n")
    sys.exit(2)
print(f"VERSION {pyarrow.__version__}")
print(f"PY_INCLUDE {sysconfig.get_path('include')}")
print(f"PA_INCLUDE {pyarrow.get_include()}")
for d in pyarrow.get_library_dirs():
    print(f"LIBDIR {d}")
print("END")
"#;
    let out = Command::new(&python)
        .args(["-c", script])
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke build-time python `{python}`: {e}"));
    if !out.status.success() {
        panic!(
            "build-time discovery via `{python}` failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut paths = DiscoveredPaths {
        pyarrow_includes: Vec::new(),
        pyarrow_libdirs: Vec::new(),
        python_include: String::new(),
        pyarrow_version: String::from("unknown"),
    };
    for line in text.lines() {
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix("VERSION ") {
            paths.pyarrow_version = rest.to_string();
        } else if let Some(rest) = line.strip_prefix("PY_INCLUDE ") {
            paths.python_include = rest.to_string();
        } else if let Some(rest) = line.strip_prefix("PA_INCLUDE ") {
            paths.pyarrow_includes.push(rest.to_string());
        } else if let Some(rest) = line.strip_prefix("LIBDIR ") {
            paths.pyarrow_libdirs.push(rest.to_string());
        }
    }
    println!(
        "cargo:warning=daft-parquet-bloom: building against pyarrow {}",
        paths.pyarrow_version
    );
    paths
}

fn main() {
    println!("cargo:rerun-if-changed=shim/shim.cpp");
    println!("cargo:rerun-if-changed=shim/shim.h");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    let paths = discover_paths();

    let mut build = cc::Build::new();
    build.cpp(true).file("shim/shim.cpp").std("c++20");

    for inc in &paths.pyarrow_includes {
        build.include(inc);
    }
    if !paths.python_include.is_empty() {
        build.include(&paths.python_include);
    }

    // On Linux, arrow-cpp is built with -D_GLIBCXX_USE_CXX11_ABI=0 to match
    // manylinux pre-CXX11 ABI. PyArrow honors the same; we mirror it.
    if cfg!(target_os = "linux") {
        build.define("_GLIBCXX_USE_CXX11_ABI", "0");
    }

    build.compile("daft_parquet_bloom_shim");

    // Link search paths
    for dir in &paths.pyarrow_libdirs {
        println!("cargo:rustc-link-search=native={dir}");
    }
    // parquet: the bloom-filter API we are calling.
    // arrow: parquet links it transitively (BufferReader, RandomAccessFile).
    // arrow_python: provides arrow::py::unwrap_random_access_file so the
    // shim can accept a pyarrow.NativeFile PyObject directly (remote files
    // become transparent — pyiceberg's FileIO hands us a NativeFile).
    let required = ["parquet", "arrow", "arrow_python"];
    for lib in &required {
        // PyArrow library names on Linux include a version suffix in the
        // SONAME, but the import name (passed to -l) is the unversioned
        // form because PyArrow ships a symlink. macOS dylib names are
        // already canonical. On Windows the .lib import file is unversioned.
        println!("cargo:rustc-link-lib=dylib={lib}");
    }

    // Inform consumers of this crate where pyarrow lives, in case they
    // need to construct an rpath at link time. Not currently consumed.
    println!(
        "cargo:pyarrow-libdir={}",
        paths.pyarrow_libdirs.first().cloned().unwrap_or_default()
    );

    let _ = PathBuf::new(); // silence the unused std::path import
}
