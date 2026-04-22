use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use cas::fs::FsCas;
use fetch::model_importert::ModelImporter;
use reqwest::Url;
use tensorfs::cas::Cas;
use tensorfs::error::TensorFsError;
use tensorfs::safetensors::{TensorMeta, parse_header};
use tensorfs::source::{RemoteFile, RemoteSnapshot, RemoteSource};

#[derive(Clone, Debug, PartialEq, Eq)]
struct RangeCall {
    path: String,
    offset: u64,
    len: u64,
}

#[derive(Clone)]
struct FakeRemote {
    files: BTreeMap<String, Vec<u8>>,
    range_calls: Arc<Mutex<Vec<RangeCall>>>,
    header_calls: Arc<Mutex<Vec<String>>>,
}

impl FakeRemote {
    fn new(files: BTreeMap<String, Vec<u8>>) -> Self {
        Self {
            files,
            range_calls: Arc::new(Mutex::new(Vec::new())),
            header_calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn url(path: &str) -> Url {
        Url::parse(&format!("https://fake.local/{path}")).unwrap()
    }

    fn path(url: &Url) -> String {
        url.path().trim_start_matches('/').to_string()
    }

    fn range_calls(&self) -> Vec<RangeCall> {
        self.range_calls.lock().unwrap().clone()
    }

    fn header_calls(&self) -> Vec<String> {
        self.header_calls.lock().unwrap().clone()
    }
}

impl RemoteSource for FakeRemote {
    async fn get_snapshot(
        &self,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<RemoteSnapshot, TensorFsError> {
        let files = self
            .files
            .iter()
            .map(|(path, bytes)| RemoteFile {
                path: path.clone(),
                size: bytes.len() as u64,
                url: Self::url(path),
            })
            .collect();

        Ok(RemoteSnapshot {
            id: model_id.to_string(),
            revision: revision.unwrap_or("test-revision").to_string(),
            files,
        })
    }

    async fn fetch_range(&self, url: &Url, offset: u64, len: u64) -> Result<Bytes, TensorFsError> {
        let path = Self::path(url);
        self.range_calls.lock().unwrap().push(RangeCall {
            path: path.clone(),
            offset,
            len,
        });

        let file = self.files.get(&path).ok_or(TensorFsError::NotFound)?;
        let start = usize::try_from(offset).map_err(|_| TensorFsError::InvalidArgument)?;
        let len = usize::try_from(len).map_err(|_| TensorFsError::InvalidArgument)?;
        let end = start
            .checked_add(len)
            .ok_or(TensorFsError::InvalidArgument)?;
        let range = file.get(start..end).ok_or(TensorFsError::InvalidArgument)?;

        Ok(Bytes::copy_from_slice(range))
    }

    async fn fetch_safetensors_header(&self, url: &Url) -> Result<Vec<TensorMeta>, TensorFsError> {
        let path = Self::path(url);
        self.header_calls.lock().unwrap().push(path.clone());
        let file = self.files.get(&path).ok_or(TensorFsError::NotFound)?;

        parse_header(file)
    }
}

fn tiny_safetensors() -> Vec<u8> {
    let json = r#"{"weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(json.as_bytes());
    bytes.extend_from_slice(&[1, 2, 3, 4]);
    bytes
}

fn fake_files() -> BTreeMap<String, Vec<u8>> {
    BTreeMap::from([
        (
            "config.json".to_string(),
            br#"{"model_type":"tiny"}"#.to_vec(),
        ),
        ("weights.safetensors".to_string(), tiny_safetensors()),
    ])
}

#[tokio::test]
async fn model_importer_downloads_ranges_into_cas() {
    let temp = tempfile::tempdir().unwrap();
    let cas_dir = temp.path().join("cas");
    let files = fake_files();
    let remote = FakeRemote::new(files.clone());
    let storage = FsCas::new(cas_dir.clone());
    let importer = ModelImporter::new(remote.clone(), storage);

    let snapshot = importer.snapshot("Qwen/tiny-test", None).await.unwrap();
    let manifest = importer.download(snapshot, |_| Ok(())).await.unwrap();

    assert_eq!(remote.header_calls(), vec!["weights.safetensors"]);
    assert!(
        remote
            .range_calls()
            .iter()
            .any(|call| call.path == "config.json" && call.offset == 0)
    );
    assert!(
        remote
            .range_calls()
            .iter()
            .any(|call| call.path == "weights.safetensors" && call.offset == 0)
    );

    let cas = FsCas::new(cas_dir);
    let zero_chunk_id = "0".repeat(64);

    for file in &manifest.files {
        let expected = files.get(&file.path).unwrap();
        let mut reconstructed = Vec::new();

        for segment in &file.segments {
            assert_ne!(segment.chunk_id.to_hex(), zero_chunk_id);
            assert!(cas.exists(segment.chunk_id).await.unwrap());

            let chunk = cas.get(segment.chunk_id).await.unwrap();
            assert_eq!(chunk.len() as u64, segment.len);
            reconstructed.extend_from_slice(&chunk);
        }

        assert_eq!(&reconstructed, expected);
    }
}
