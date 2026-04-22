use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use api::prefetch_service::PrefetchService;
use bytes::Bytes;
use cas::fs::FsCas;
use core::cas::Cas;
use core::chunk::ChunkId;
use core::error::TensorFsError;
use core::manifest::Manifest;
use core::safetensors::{TensorMeta, parse_header};
use core::source::{RemoteFile, RemoteSnapshot, RemoteSource};
use fetch::model_importert::ModelImporter;
use url::Url;

#[derive(Clone)]
struct FakeRemote {
    files: BTreeMap<String, Vec<u8>>,
    range_calls: Arc<Mutex<Vec<RangeCall>>>,
    fail_after_successful_ranges: Arc<Mutex<Option<usize>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RangeCall {
    path: String,
    offset: u64,
    len: u64,
}

impl FakeRemote {
    fn new(files: BTreeMap<String, Vec<u8>>) -> Self {
        Self {
            files,
            range_calls: Arc::new(Mutex::new(Vec::new())),
            fail_after_successful_ranges: Arc::new(Mutex::new(None)),
        }
    }

    fn with_fail_after_successful_ranges(files: BTreeMap<String, Vec<u8>>, limit: usize) -> Self {
        let remote = Self::new(files);
        *remote.fail_after_successful_ranges.lock().unwrap() = Some(limit);
        remote
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

        {
            let range_calls = self.range_calls.lock().unwrap();
            let limit = *self.fail_after_successful_ranges.lock().unwrap();
            if limit.is_some_and(|limit| range_calls.len() >= limit) {
                return Err(TensorFsError::InvalidResponse);
            }
        }

        let file = self.files.get(&path).ok_or(TensorFsError::NotFound)?;
        let start = usize::try_from(offset).map_err(|_| TensorFsError::InvalidArgument)?;
        let len = usize::try_from(len).map_err(|_| TensorFsError::InvalidArgument)?;
        let end = start
            .checked_add(len)
            .ok_or(TensorFsError::InvalidArgument)?;
        let range = file.get(start..end).ok_or(TensorFsError::InvalidArgument)?;

        self.range_calls.lock().unwrap().push(RangeCall {
            path,
            offset,
            len: len as u64,
        });

        Ok(Bytes::copy_from_slice(range))
    }

    async fn fetch_safetensors_header(&self, url: &Url) -> Result<Vec<TensorMeta>, TensorFsError> {
        let path = Self::path(url);
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

fn make_service(
    files: BTreeMap<String, Vec<u8>>,
    cas_dir: &Path,
    manifest_dir: &Path,
) -> PrefetchService<FakeRemote, FsCas> {
    make_service_with_remote(FakeRemote::new(files), cas_dir, manifest_dir)
}

fn make_service_with_remote(
    remote: FakeRemote,
    cas_dir: &Path,
    manifest_dir: &Path,
) -> PrefetchService<FakeRemote, FsCas> {
    let storage = FsCas::new(cas_dir.to_path_buf());
    let importer = ModelImporter::new(remote, storage);

    PrefetchService::new(importer, manifest_dir.to_path_buf())
}

async fn assert_manifest_reconstructs_files(
    manifest: &Manifest,
    files: &BTreeMap<String, Vec<u8>>,
    cas_dir: &Path,
) {
    let cas = FsCas::new(cas_dir.to_path_buf());
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

fn manifest_signature(manifest: &Manifest) -> Vec<(String, u64, Vec<(String, u64, u64, u64)>)> {
    manifest
        .files
        .iter()
        .map(|file| {
            let segments = file
                .segments
                .iter()
                .map(|segment| {
                    (
                        segment.chunk_id.to_hex(),
                        segment.file_offset,
                        segment.chunk_offset,
                        segment.len,
                    )
                })
                .collect();

            (file.path.clone(), file.size, segments)
        })
        .collect()
}

fn count_cas_chunks(cas_dir: &Path) -> usize {
    fs::read_dir(cas_dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|file_type| file_type.is_file()))
        .count()
}

fn count_present_segments(manifest: &Manifest) -> usize {
    manifest
        .files
        .iter()
        .flat_map(|file| &file.segments)
        .filter(|segment| !segment.chunk_id.is_empty())
        .count()
}

fn count_segments(manifest: &Manifest) -> usize {
    manifest.files.iter().map(|file| file.segments.len()).sum()
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(future)
}

#[test]
fn prefetch_imports_to_cas_and_saves_manifest() {
    block_on(async {
        let temp = tempfile::tempdir().unwrap();
        let cas_dir = temp.path().join("cas");
        let manifest_dir = temp.path().join("manifests");
        let files = fake_files();
        let service = make_service(files.clone(), &cas_dir, &manifest_dir);

        let manifest = service.prefetch("Qwen/tiny-test").await.unwrap();

        assert_eq!(manifest.source, "hf://Qwen/tiny-test");
        assert_eq!(manifest.files.len(), 2);

        let saved_path = manifest_dir.join("Qwen/tiny-test");
        assert!(saved_path.exists());

        let saved = Manifest::load(&saved_path).unwrap();
        assert_eq!(manifest_signature(&saved), manifest_signature(&manifest));

        assert_manifest_reconstructs_files(&saved, &files, &cas_dir).await;

        let safetensors = saved
            .files
            .iter()
            .find(|file| file.path == "weights.safetensors")
            .unwrap();
        assert_eq!(safetensors.segments.len(), 2);
    });
}

#[test]
fn prefetch_is_idempotent() {
    block_on(async {
        let temp = tempfile::tempdir().unwrap();
        let cas_dir = temp.path().join("cas");
        let manifest_dir = temp.path().join("manifests");
        let files = fake_files();
        let service = make_service(files.clone(), &cas_dir, &manifest_dir);

        let first = service.prefetch("Qwen/tiny-test").await.unwrap();
        let cas_count_after_first = count_cas_chunks(&cas_dir);

        let second = service.prefetch("Qwen/tiny-test").await.unwrap();
        let cas_count_after_second = count_cas_chunks(&cas_dir);

        assert_eq!(manifest_signature(&first), manifest_signature(&second));
        assert_eq!(cas_count_after_first, cas_count_after_second);

        let saved = Manifest::load(&manifest_dir.join("Qwen/tiny-test")).unwrap();
        assert_eq!(manifest_signature(&saved), manifest_signature(&second));
        assert_manifest_reconstructs_files(&saved, &files, &cas_dir).await;
    });
}

#[test]
fn prefetch_repairs_missing_cas_chunk() {
    block_on(async {
        let temp = tempfile::tempdir().unwrap();
        let cas_dir = temp.path().join("cas");
        let manifest_dir = temp.path().join("manifests");
        let files = fake_files();
        let service = make_service(files.clone(), &cas_dir, &manifest_dir);

        let first = service.prefetch("Qwen/tiny-test").await.unwrap();
        let deleted_chunk_id: ChunkId = first.files[0].segments[0].chunk_id;
        let deleted_chunk_path = cas_dir.join(deleted_chunk_id.to_hex());
        fs::remove_file(&deleted_chunk_path).unwrap();
        assert!(!deleted_chunk_path.exists());

        let repaired = service.prefetch("Qwen/tiny-test").await.unwrap();

        assert!(deleted_chunk_path.exists());
        assert_eq!(manifest_signature(&first), manifest_signature(&repaired));
        assert_manifest_reconstructs_files(&repaired, &files, &cas_dir).await;
    });
}

#[test]
fn prefetch_resumes_from_progress_manifest_after_interrupted_download() {
    block_on(async {
        let temp = tempfile::tempdir().unwrap();
        let cas_dir = temp.path().join("cas");
        let manifest_dir = temp.path().join("manifests");
        let files = fake_files();
        let failing_remote = FakeRemote::with_fail_after_successful_ranges(files.clone(), 1);
        let service = make_service_with_remote(failing_remote.clone(), &cas_dir, &manifest_dir);

        let err = service.prefetch("Qwen/tiny-test").await.unwrap_err();

        assert!(matches!(err, TensorFsError::InvalidResponse));
        let initial_calls = failing_remote.range_calls();
        assert_eq!(initial_calls.len(), 1);

        let saved_path = manifest_dir.join("Qwen/tiny-test");
        assert!(saved_path.exists());
        let partial = Manifest::load(&saved_path).unwrap();
        assert_eq!(count_present_segments(&partial), 1);

        let resume_remote = FakeRemote::new(files.clone());
        let service = make_service_with_remote(resume_remote.clone(), &cas_dir, &manifest_dir);

        let resumed = service.prefetch("Qwen/tiny-test").await.unwrap();

        let resume_calls = resume_remote.range_calls();
        assert_eq!(resume_calls.len(), count_segments(&resumed) - 1);
        assert!(!resume_calls.iter().any(|call| call == &initial_calls[0]));

        let saved = Manifest::load(&saved_path).unwrap();
        assert_eq!(manifest_signature(&saved), manifest_signature(&resumed));
        assert_manifest_reconstructs_files(&resumed, &files, &cas_dir).await;
    });
}

#[test]
fn prefetch_rejects_unsafe_model_id() {
    block_on(async {
        let temp = tempfile::tempdir().unwrap();
        let cas_dir = temp.path().join("cas");
        let manifest_dir = temp.path().join("manifests");
        let service = make_service(fake_files(), &cas_dir, &manifest_dir);

        let err = service.prefetch("../outside").await.unwrap_err();

        assert!(matches!(err, TensorFsError::InvalidArgument));
        assert!(!temp.path().join("outside").exists());
        assert!(!cas_dir.exists());
    });
}
