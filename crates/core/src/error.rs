use std::io;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, TensorFsError>;

#[derive(Error, Debug)]
pub enum TensorFsError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("Invalid hex")]
    InvalidHex,
    #[error("Invalid chunk_id length")]
    InvalidChunkIdLength,
}
