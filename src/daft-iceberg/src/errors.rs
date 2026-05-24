use common_error::DaftError;

#[derive(Debug, thiserror::Error)]
pub enum IcebergRewriteError {
    #[error("invalid option `{name}`: {reason}")]
    InvalidOption { name: String, reason: String },

    #[error(
        "equality deletes present in {n} file(s); rewrite_data_files requires equality \
         deletes to be applied first (e.g. via rewrite_position_delete_files). Files: {sample:?}"
    )]
    EqualityDeletesPresent { n: usize, sample: Vec<String> },

    #[error("zorder column `{column}` has unsupported type `{dtype}`")]
    UnsupportedZOrderType { column: String, dtype: String },

    #[error("output_spec_id {output} not present in table specs")]
    UnknownOutputSpec { output: i32 },
}

impl From<IcebergRewriteError> for DaftError {
    fn from(e: IcebergRewriteError) -> Self {
        DaftError::ValueError(e.to_string())
    }
}
