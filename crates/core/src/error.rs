use std::io;

use thiserror::Error;

use crate::chunk;

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
    #[error("Segment not found")]
    SegmentNotFound,
    #[error("Resolver out of bound")]
    ResolverOutOfBound,
    #[error("Incorrect read interval")]
    IncorrectReadInterval,
    #[error("Safetensors incorrect header length")]
    IncorrectSafetensorsLen,
    #[error("Safetensors header read error")]
    SafeTensorsReadError,
    #[error("Header len overflow")]
    HeaderLenOverflow,
    #[error("Incomplete header")]
    IncompleteHeader,
    #[error("Invalid header json")]
    InvalidHeaderJson,
    #[error("Offset overflow")]
    OffsetOverflow,
    #[error("Invalid offset")]
    InvalidOffsets,
    #[error("Invalid tensor size")]
    InvalidTensorSize,
    #[error("Unauthorized")]
    Unauthorized,
    #[error("Bad request")]
    BadRequest,
    #[error("Invalid json")]
    InvalidJson,
    #[error("Forbidden")]
    Forbidden,
    #[error("Range not satisfiable")]
    RangeNotSatisfiable,
    #[error("Invalid response")]
    InvalidResponse,
    #[error("Not found")]
    NotFound,
}
