use bytes::Bytes;
use url::Url;

use crate::error::TensorFsError;
use crate::safetensors::TensorMeta;

pub trait RemoteSource: Send + Sync {
    fn list_model_files(
        &self,
        model_id: &str,
    ) -> impl Future<Output = Result<Vec<RemoteFile>, TensorFsError>> + Send;
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
