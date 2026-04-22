use bytes::Bytes;
use url::Url;

use crate::error::TensorFsError;
use crate::safetensors::TensorMeta;

pub trait RemoteSource: Send + Sync {
    fn get_snapshot(
        &self,
        model_id: &str,
        revision: Option<&str>,
    ) -> impl Future<Output = Result<RemoteSnapshot, TensorFsError>> + Send;
    fn fetch_range(
        &self,
        url: &Url,
        offset: u64,
        len: u64,
    ) -> impl Future<Output = Result<Bytes, TensorFsError>> + Send;
    fn fetch_safetensors_header(
        &self,
        url: &Url,
    ) -> impl Future<Output = Result<Vec<TensorMeta>, TensorFsError>> + Send;
}

#[derive(Clone)]
pub struct RemoteFile {
    pub path: String,
    pub size: u64,
    pub url: Url,
}

#[derive(Clone)]
pub struct RemoteSnapshot {
    pub id: String,
    pub revision: String,
    pub files: Vec<RemoteFile>,
}
