use fuser::Errno;
use libc;
use tensorfs::error::TensorFsError;

pub fn to_errno(err: TensorFsError) -> Errno {
    let code = match err {
        TensorFsError::Io(e) => e.raw_os_error().unwrap_or(libc::EIO),

        TensorFsError::Unauthorized | TensorFsError::Forbidden => libc::EACCES,

        TensorFsError::ChunkNotFound(_)
        | TensorFsError::SegmentNotFound
        | TensorFsError::NotFound => libc::ENOENT,

        TensorFsError::InvalidArgument
        | TensorFsError::InvalidHex
        | TensorFsError::InvalidChunkIdLength
        | TensorFsError::ValidationError
        | TensorFsError::InvalidJson
        | TensorFsError::BadRequest
        | TensorFsError::IncorrectReadInterval
        | TensorFsError::RangeNotSatisfiable => libc::EINVAL,

        TensorFsError::ManifestReadError
        | TensorFsError::ManifestWriteError
        | TensorFsError::SafeTensorsReadError
        | TensorFsError::InvalidHeaderJson
        | TensorFsError::IncompleteHeader
        | TensorFsError::IncorrectSafetensorsLen => libc::EIO,

        TensorFsError::ResolverOutOfBound
        | TensorFsError::OffsetOverflow
        | TensorFsError::InvalidOffsets
        | TensorFsError::HeaderLenOverflow
        | TensorFsError::InvalidTensorSize => libc::EOVERFLOW,

        TensorFsError::ManifestValidationError | TensorFsError::InvalidResponse => libc::EBADMSG,
    };

    Errno::from_i32(code)
}
