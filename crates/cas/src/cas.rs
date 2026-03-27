use std::io::SeekFrom;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use tensorfs::error;
use tensorfs::chunk;
use tokio::fs::{OpenOptions, create_dir_all, read, try_exists, remove_file, rename};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

pub trait Cas: Send + Sync {
    fn put(&self, bytes: Bytes) -> impl Future<Output = Result<chunk::ChunkId, error::TensorFsError>> + Send;
    fn get(&self, id: chunk::ChunkId) -> impl Future<Output = Result<Bytes, error::TensorFsError>> + Send;
    fn exists(&self, id: chunk::ChunkId) -> impl Future<Output = Result<bool, error::TensorFsError>> + Send;
    fn read_range(&self, id: chunk::ChunkId, offset: u64, len: usize) -> impl Future<Output = Result<Bytes, error::TensorFsError>> + Send;
}

pub struct FsCas {
    path: PathBuf,
}

impl Cas for FsCas {
    async fn put(&self, bytes: Bytes) -> Result<chunk::ChunkId, error::TensorFsError> {
        let hash = blake3::hash(&bytes);
        let id = chunk::ChunkId::from_bytes(*hash.as_bytes());

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

    async fn get(&self, id: chunk::ChunkId) -> Result<Bytes, error::TensorFsError> {
        let path = std::path::Path::new(&self.path).join(id.to_hex());
        let file = read(path).await?;
        Ok(Bytes::from(file))
    }

    async fn exists(&self, id: chunk::ChunkId) -> Result<bool, error::TensorFsError> {
        let path = std::path::Path::new(&self.path).join(id.to_hex());
        let exists = try_exists(path).await?;
        Ok(exists)
    }

    async fn read_range(&self, id: chunk::ChunkId, offset: u64, len: usize) -> Result<Bytes, error::TensorFsError> {
        if len == 0 {
            return Ok(Bytes::new());
        }

        let path = std::path::Path::new(&self.path).join(id.to_hex());

        let mut f = OpenOptions::new().read(true).open(path).await?;

        f.seek(SeekFrom::Start(offset)).await?;

        let mut buffer = vec![0u8; len];
        let got = f.read(&mut buffer).await?;

        buffer.truncate(got);

        Ok(Bytes::from(buffer))
    }
}

impl FsCas {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn chunk_path(&self, id: &chunk::ChunkId) -> PathBuf {
        Path::new(&self.path).join(id.to_hex())
    }

    fn temp_chunk_path(&self, id: &chunk::ChunkId) -> PathBuf {
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
    use bytes::Bytes;
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
        let expected = Bytes::from_static(b"hello cas");

        let id = cas.put(expected.clone()).await.unwrap();
        let actual = cas.get(id).await.unwrap();

        assert_eq!(actual, expected);

        let _ = tokio::fs::remove_dir_all(dir).await;
    }

    #[tokio::test]
    async fn exists_reports_missing_and_present_chunks() {
        let dir = test_dir("exists");
        let cas = FsCas::new(dir.clone());

        let missing_id = tensorfs::chunk::ChunkId::from_bytes(*blake3::hash(b"missing").as_bytes());
        assert!(!cas.exists(missing_id).await.unwrap());

        let id = cas.put(Bytes::from_static(b"present")).await.unwrap();
        assert!(cas.exists(id).await.unwrap());

        let _ = tokio::fs::remove_dir_all(dir).await;
    }

    #[tokio::test]
    async fn put_does_not_create_duplicate_files_for_same_bytes() {
        let dir = test_dir("dedupe");
        let cas = FsCas::new(dir.clone());
        let bytes = Bytes::from_static(b"same bytes");

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

    #[tokio::test]
    async fn read_range_reads_middle_of_chunk() {
        let dir = test_dir("read-range-middle");
        let cas = FsCas::new(dir.clone());

        let id = cas.put(Bytes::from_static(b"hello world")).await.unwrap();
        let actual = cas.read_range(id, 6, 5).await.unwrap();

        assert_eq!(actual, Bytes::from_static(b"world"));

        let _ = tokio::fs::remove_dir_all(dir).await;
    }

    #[tokio::test]
    async fn read_range_returns_empty_when_offset_equals_size() {
        let dir = test_dir("read-range-offset-eq-size");
        let cas = FsCas::new(dir.clone());

        let content = Bytes::from_static(b"abcdef");
        let id = cas.put(content.clone()).await.unwrap();

        let actual = cas.read_range(id, content.len() as u64, 10).await.unwrap();

        assert!(actual.is_empty());

        let _ = tokio::fs::remove_dir_all(dir).await;
    }

    #[tokio::test]
    async fn read_range_returns_empty_when_offset_is_past_end() {
        let dir = test_dir("read-range-offset-past-end");
        let cas = FsCas::new(dir.clone());

        let id = cas.put(Bytes::from_static(b"abcdef")).await.unwrap();
        let actual = cas.read_range(id, 100, 10).await.unwrap();

        assert!(actual.is_empty());

        let _ = tokio::fs::remove_dir_all(dir).await;
    }

    #[tokio::test]
    async fn read_range_truncates_when_len_goes_past_end() {
        let dir = test_dir("read-range-truncate");
        let cas = FsCas::new(dir.clone());

        let id = cas.put(Bytes::from_static(b"abcdef")).await.unwrap();
        let actual = cas.read_range(id, 4, 10).await.unwrap();

        assert_eq!(actual, Bytes::from_static(b"ef"));

        let _ = tokio::fs::remove_dir_all(dir).await;
    }

    #[tokio::test]
    async fn read_range_reads_entire_chunk_when_requested() {
        let dir = test_dir("read-range-full");
        let cas = FsCas::new(dir.clone());

        let expected = Bytes::from_static(b"full chunk");
        let id = cas.put(expected.clone()).await.unwrap();

        let actual = cas.read_range(id, 0, expected.len()).await.unwrap();

        assert_eq!(actual, expected);

        let _ = tokio::fs::remove_dir_all(dir).await;
    }
}
