use bytes::Bytes;

use crate::chunk::ChunkId;
use crate::error::TensorFsError;

pub trait Cas: Send + Sync {
    fn put(&self, bytes: Bytes) -> impl Future<Output = Result<ChunkId, TensorFsError>> + Send;
    fn get(&self, id: ChunkId) -> impl Future<Output = Result<Bytes, TensorFsError>> + Send;
    fn exists(&self, id: ChunkId) -> impl Future<Output = Result<bool, TensorFsError>> + Send;
    fn read_range(
        &self,
        id: ChunkId,
        offset: u64,
        len: usize,
    ) -> impl Future<Output = Result<Bytes, TensorFsError>> + Send;
}
