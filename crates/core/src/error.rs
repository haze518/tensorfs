use std::io;

use thiserror::Error;

use crate::chunk;

pub type Result<T> = std::result::Result<T, TensorFsError>;

#[derive(Error, Debug)]
pub enum TensorFsError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("Invalid hex")]
    InvalidHex,
    #[error("Invalid chunk_id length")]
    InvalidChunkIdLength,
    #[error("No chunk with id: {0}")]
    ChunkNotFound(chunk::ChunkId),
    #[error("Invalid argument")]
    InvalidArgument,
    #[error("Manifest read error")]
    ManifestReadError,
    #[error("Manifest validation error")]
    ManifestValidationError,
}
