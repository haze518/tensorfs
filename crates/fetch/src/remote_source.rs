use bytes::Bytes;
use reqwest::{Client, RequestBuilder, StatusCode, Url};
use serde::Deserialize;
use tensorfs::error::TensorFsError;
use tensorfs::safetensors::{SAFETENSORS_HEADER_LEN, TensorMeta, parse_header};

pub struct RemoteFile {
    pub path: String,
    pub size: Option<u64>,
    pub url: Url,
}

#[derive(Deserialize)]
struct ModelResponse {
    siblings: Vec<Sibling>,
    sha: String,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
    size: Option<u64>,
}

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

pub struct HFClient {
    base_url: Url,
    token: Option<String>,
    client: Client,
}

impl RemoteSource for HFClient {
    async fn list_model_files(&self, model_id: &str) -> Result<Vec<RemoteFile>, TensorFsError> {
        let model = format!("api/models/{model_id}");
        let url = self
            .base_url
            .join(&model)
            .map_err(|_| TensorFsError::InvalidArgument)?;
        let response = self
            .build_request(&url)
            .send()
            .await
            .map_err(|_| TensorFsError::BadRequest)?;

        let status = response.status();
        let raw = match status.is_success() {
            true => Ok(response),
            false => match status {
                StatusCode::NOT_FOUND => Err(TensorFsError::NotFound),
                StatusCode::UNAUTHORIZED => Err(TensorFsError::Unauthorized),
                StatusCode::FORBIDDEN => Err(TensorFsError::Forbidden),
                _ => Err(TensorFsError::BadRequest),
            },
        }?;

        let resp: ModelResponse = raw.json().await.map_err(|_| TensorFsError::InvalidJson)?;

        let mut result = Vec::with_capacity(resp.siblings.len());
        for sib in resp.siblings {
            let model_path = format!("{model_id}/resolve/{}/{}", &resp.sha, &sib.rfilename);
            let url = self
                .base_url
                .join(&model_path)
                .map_err(|_| TensorFsError::InvalidArgument)?;
            result.push(RemoteFile {
                path: sib.rfilename,
                size: sib.size,
                url,
            });
        }

        Ok(result)
    }

    async fn fetch_range(&self, url: &Url, offset: u64, len: u64) -> Result<Bytes, TensorFsError> {
        if len == 0 {
            return Err(TensorFsError::InvalidArgument);
        }

        let end = offset
            .checked_add(len - 1)
            .ok_or(TensorFsError::IncorrectReadInterval)?;

        let response = self
            .build_request(url)
            .header("Range", format!("bytes={offset}-{end}"))
            .send()
            .await
            .map_err(|_| TensorFsError::BadRequest)?;

        let status = response.status();
        match status {
            StatusCode::PARTIAL_CONTENT => {
                let bytes = response
                    .bytes()
                    .await
                    .map_err(|_| TensorFsError::InvalidResponse)?;
                if bytes.len() != len as usize {
                    return Err(TensorFsError::InvalidResponse);
                }
                Ok(bytes)
            }
            StatusCode::OK => {
                let bytes = response
                    .bytes()
                    .await
                    .map_err(|_| TensorFsError::InvalidResponse)?;
                if bytes.len() < (offset + len) as usize {
                    return Err(TensorFsError::InvalidResponse);
                }
                let start = offset as usize;
                let end = start + len as usize;
                Ok(bytes.slice(start..end))
            }
            StatusCode::RANGE_NOT_SATISFIABLE => Err(TensorFsError::RangeNotSatisfiable),
            StatusCode::UNAUTHORIZED => Err(TensorFsError::Unauthorized),
            StatusCode::FORBIDDEN => Err(TensorFsError::Forbidden),
            StatusCode::NOT_FOUND => Err(TensorFsError::NotFound),
            _ => Err(TensorFsError::BadRequest),
        }
    }

    async fn fetch_safetensors_header(&self, url: &Url) -> Result<Vec<TensorMeta>, TensorFsError> {
        let prefix = self
            .fetch_range(&url, 0, SAFETENSORS_HEADER_LEN as u64)
            .await?;

        if prefix.len() < SAFETENSORS_HEADER_LEN {
            return Err(TensorFsError::IncorrectSafetensorsLen);
        }

        let header_len = u64::from_le_bytes(
            prefix[..SAFETENSORS_HEADER_LEN]
                .try_into()
                .map_err(|_| TensorFsError::IncorrectSafetensorsLen)?,
        );

        let header = self
            .fetch_range(&url, SAFETENSORS_HEADER_LEN as u64, header_len)
            .await?;

        if header.len() as u64 != header_len {
            return Err(TensorFsError::IncorrectSafetensorsLen)
        }

        let mut buf = Vec::with_capacity(SAFETENSORS_HEADER_LEN + header.len());
        buf.extend_from_slice(&prefix);
        buf.extend_from_slice(&header);

        parse_header(&buf)
    }
}

impl HFClient {
    pub fn new(base_url: Url, token: Option<String>) -> Self {
        Self {
            base_url,
            token,
            client: Client::new(),
        }
    }

    fn build_request(&self, url: &Url) -> RequestBuilder {
        let req = self.client.get(url.clone());

        if let Some(token) = &self.token {
            req.bearer_auth(token)
        } else {
            req
        }
    }
}
