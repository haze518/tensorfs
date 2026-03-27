use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use tensorfs::error;
use tensorfs::types;
use tokio::fs::{OpenOptions, create_dir_all, read, try_exists, remove_file, rename};
use tokio::io::AsyncWriteExt;

pub trait Cas: Send + Sync {
    fn put(&self, bytes: bytes::Bytes) -> impl Future<Output = Result<types::ChunkId, error::TensorFsError>>;
    fn get(&self, id: types::ChunkId) -> impl Future<Output = Result<bytes::Bytes, error::TensorFsError>> + Send;
    fn exists(&self, id: types::ChunkId) -> impl Future<Output = Result<bool, error::TensorFsError>>;
}

pub struct FsCas {
    path: PathBuf,
}

impl Cas for FsCas {
    async fn put(&self, bytes: bytes::Bytes) -> Result<types::ChunkId, error::TensorFsError> {
        let hash = blake3::hash(&bytes);
        let id = types::ChunkId::from_bytes(*hash.as_bytes());

        create_dir_all(&self.path).await?;

        let final_path = self.chunk_path(&id);

        match tokio::fs::try_exists(&final_path).await {
            Ok(true) => return Ok(id),
            Ok(false) => {}
            Err(e) => return Err(e.into()),
        }

        let tmp_path = self.temp_chunk_path(&id);

        let mut file = match OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp_path)
            .await
        {
            Ok(file) => file,
            Err(e) => return Err(e.into()),
        };

        let write_result = async {
            file.write_all(&bytes).await?;
            file.sync_all().await?;
            Ok::<(), std::io::Error>(())
        }
        .await;

        if let Err(e) = write_result {
            let _ = remove_file(&tmp_path).await;
            return Err(e.into());
        }

        match rename(&tmp_path, &final_path).await {
            Ok(()) => Ok(id),

            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let _ = remove_file(&tmp_path).await;
                Ok(id)
            }

            Err(e) => {
                let _ = remove_file(&tmp_path).await;
                Err(e.into())
            }
        }
    }

    async fn get(&self, id: types::ChunkId) -> Result<bytes::Bytes, error::TensorFsError> {
        let path = std::path::Path::new(&self.path).join(id.to_hex());
        let file = read(path).await?;
        Ok(bytes::Bytes::from(file))
    }

    async fn exists(&self, id: types::ChunkId) -> Result<bool, error::TensorFsError> {
        let path = std::path::Path::new(&self.path).join(id.to_hex());
        let exists = try_exists(path).await?;
        Ok(exists)
    }
}

impl FsCas {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn chunk_path(&self, id: &types::ChunkId) -> PathBuf {
        Path::new(&self.path).join(id.to_hex())
    }

    fn temp_chunk_path(&self, id: &types::ChunkId) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        Path::new(&self.path).join(format!("{}.tmp-{nanos}", id.to_hex()))
    }
}

impl Default for FsCas {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./cas"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cas, FsCas};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        std::env::temp_dir().join(format!(
            "tensorfs-cas-test-{name}-{}-{nanos}",
            std::process::id()
        ))
    }

    #[tokio::test]
    async fn put_then_get_returns_same_bytes() {
        let dir = test_dir("put-get");
        let cas = FsCas::new(dir.clone());
        let expected = bytes::Bytes::from_static(b"hello cas");

        let id = cas.put(expected.clone()).await.unwrap();
        let actual = cas.get(id).await.unwrap();

        assert_eq!(actual, expected);

        let _ = tokio::fs::remove_dir_all(dir).await;
    }

    #[tokio::test]
    async fn exists_reports_missing_and_present_chunks() {
        let dir = test_dir("exists");
        let cas = FsCas::new(dir.clone());

        let missing_id = tensorfs::types::ChunkId::from_bytes(*blake3::hash(b"missing").as_bytes());
        assert!(!cas.exists(missing_id).await.unwrap());

        let id = cas.put(bytes::Bytes::from_static(b"present")).await.unwrap();
        assert!(cas.exists(id).await.unwrap());

        let _ = tokio::fs::remove_dir_all(dir).await;
    }

    #[tokio::test]
    async fn put_does_not_create_duplicate_files_for_same_bytes() {
        let dir = test_dir("dedupe");
        let cas = FsCas::new(dir.clone());
        let bytes = bytes::Bytes::from_static(b"same bytes");

        let first_id = cas.put(bytes.clone()).await.unwrap();
        let first_hex = first_id.to_hex();

        let second_id = cas.put(bytes).await.unwrap();
        let second_hex = second_id.to_hex();

        assert_eq!(first_hex, second_hex);

        let entries = std::fs::read_dir(&dir)
            .unwrap()
            .collect::<Result<Vec<_>, std::io::Error>>()
            .unwrap();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_name().to_string_lossy(), first_hex);

        let _ = tokio::fs::remove_dir_all(dir).await;
    }
}
