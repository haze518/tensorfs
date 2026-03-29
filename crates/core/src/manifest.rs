use std::fs::read;
use std::path::Path;
use std::str;

use serde::{Deserialize, Serialize};

use crate::{chunk::ChunkId, error};

/// Example JSON:
/// {
///   "version": 1,
///   "source": "...",
///   "files": [
///     {
///       "path": "...",
///       "size": 123,
///       "segments": [
///         {
///           "chunk_id": "...",
///           "file_offset": 0,
///           "chunk_offset": 0,
///           "len": 123
///         }
///       ]
///     }
///   ]
/// }
#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    version: u16,
    source: String,
    files: Vec<File>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct File {
    path: String,
    size: u64,
    segments: Vec<Segment>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Segment {
    chunk_id: ChunkId,
    file_offset: u64,
    chunk_offset: u64,
    len: u64,
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self, error::TensorFsError> {
        let data = read(path)?;

        let manifest: Manifest =
            serde_json::from_slice(&data).map_err(|_| error::TensorFsError::ManifestReadError)?;

        manifest.validate()?;

        Ok(manifest)
    }

    fn validate(&self) -> Result<(), error::TensorFsError> {
        for file in &self.files {
            let mut prev_end = 0;
            for segment in &file.segments {
                if segment.len == 0 {
                    return Err(error::TensorFsError::ManifestValidationError);
                }
                if segment.chunk_offset.checked_add(segment.len).is_none() {
                    return Err(error::TensorFsError::ManifestValidationError);
                }
                if segment.file_offset.checked_add(segment.len).is_none() {
                    return Err(error::TensorFsError::ManifestValidationError);
                }

                if segment.file_offset != prev_end {
                    return Err(error::TensorFsError::ManifestValidationError);
                }
                prev_end += segment.len
            }
            if prev_end != file.size {
                return Err(error::TensorFsError::ManifestValidationError);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file_path(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        std::env::temp_dir().join(format!(
            "tensorfs-manifest-test-{name}-{}-{nanos}.json",
            std::process::id()
        ))
    }

    fn test_chunk_id(byte: u8) -> ChunkId {
        ChunkId::from_bytes([byte; 32])
    }

    fn sample_manifest() -> Manifest {
        Manifest {
            version: 1,
            source: "hf://meta-llama/Llama-3".to_string(),
            files: vec![File {
                path: "model-00001-of-00002.safetensors".to_string(),
                size: 8,
                segments: vec![
                    Segment {
                        chunk_id: test_chunk_id(1),
                        file_offset: 0,
                        chunk_offset: 0,
                        len: 4,
                    },
                    Segment {
                        chunk_id: test_chunk_id(2),
                        file_offset: 4,
                        chunk_offset: 0,
                        len: 4,
                    },
                ],
            }],
        }
    }

    #[test]
    fn serializes_manifest_to_json() {
        let manifest = sample_manifest();

        let json = serde_json::to_string(&manifest).unwrap();

        assert!(json.contains("\"version\":1"));
        assert!(json.contains("\"source\":\"hf://meta-llama/Llama-3\""));
        assert!(json.contains("\"path\":\"model-00001-of-00002.safetensors\""));
        assert!(json.contains("\"size\":8"));
        assert!(json.contains("\"file_offset\":0"));
        assert!(json.contains("\"chunk_offset\":0"));
        assert!(json.contains("\"len\":4"));
    }

    #[test]
    fn deserializes_manifest_from_json() {
        let chunk1 = test_chunk_id(1);
        let chunk2 = test_chunk_id(2);

        let json = format!(
            r#"{{
                "version": 1,
                "source": "hf://meta-llama/Llama-3",
                "files": [
                    {{
                        "path": "model-00001-of-00002.safetensors",
                        "size": 8,
                        "segments": [
                            {{
                                "chunk_id": "{}",
                                "file_offset": 0,
                                "chunk_offset": 0,
                                "len": 4
                            }},
                            {{
                                "chunk_id": "{}",
                                "file_offset": 4,
                                "chunk_offset": 0,
                                "len": 4
                            }}
                        ]
                    }}
                ]
            }}"#,
            chunk1.to_hex(),
            chunk2.to_hex()
        );

        let manifest: Manifest = serde_json::from_str(&json).unwrap();

        assert_eq!(manifest.version, 1);
        assert_eq!(manifest.source, "hf://meta-llama/Llama-3");
        assert_eq!(manifest.files.len(), 1);

        let file = &manifest.files[0];
        assert_eq!(file.path, "model-00001-of-00002.safetensors");
        assert_eq!(file.size, 8);
        assert_eq!(file.segments.len(), 2);

        let first = &file.segments[0];
        assert_eq!(first.chunk_id, chunk1);
        assert_eq!(first.file_offset, 0);
        assert_eq!(first.chunk_offset, 0);
        assert_eq!(first.len, 4);

        let second = &file.segments[1];
        assert_eq!(second.chunk_id, chunk2);
        assert_eq!(second.file_offset, 4);
        assert_eq!(second.chunk_offset, 0);
        assert_eq!(second.len, 4);
    }

    #[test]
    fn manifest_round_trip_json_preserves_data() {
        let manifest = sample_manifest();

        let json = serde_json::to_vec(&manifest).unwrap();
        let decoded: Manifest = serde_json::from_slice(&json).unwrap();

        assert_eq!(decoded.version, manifest.version);
        assert_eq!(decoded.source, manifest.source);
        assert_eq!(decoded.files.len(), manifest.files.len());

        let expected_file = &manifest.files[0];
        let actual_file = &decoded.files[0];

        assert_eq!(actual_file.path, expected_file.path);
        assert_eq!(actual_file.size, expected_file.size);
        assert_eq!(actual_file.segments.len(), expected_file.segments.len());

        for (actual, expected) in actual_file
            .segments
            .iter()
            .zip(expected_file.segments.iter())
        {
            assert_eq!(actual.chunk_id, expected.chunk_id);
            assert_eq!(actual.file_offset, expected.file_offset);
            assert_eq!(actual.chunk_offset, expected.chunk_offset);
            assert_eq!(actual.len, expected.len);
        }
    }

    #[test]
    fn from_path_reads_and_deserializes_manifest() {
        let path = temp_file_path("from-path");
        let manifest = sample_manifest();
        let json = serde_json::to_vec_pretty(&manifest).unwrap();

        fs::write(&path, json).unwrap();

        let loaded = Manifest::load(&path).unwrap();

        assert_eq!(loaded.version, manifest.version);
        assert_eq!(loaded.source, manifest.source);
        assert_eq!(loaded.files.len(), manifest.files.len());
        assert_eq!(loaded.files[0].path, manifest.files[0].path);
        assert_eq!(loaded.files[0].size, manifest.files[0].size);
        assert_eq!(
            loaded.files[0].segments.len(),
            manifest.files[0].segments.len()
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn from_path_returns_error_for_invalid_json() {
        let path = temp_file_path("invalid-json");
        fs::write(&path, br#"{ "version": 1, "files": [ "#).unwrap();

        let err = Manifest::load(&path).unwrap_err();

        assert!(matches!(err, error::TensorFsError::ManifestReadError));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn from_path_returns_error_for_non_utf8_file() {
        let path = temp_file_path("invalid-utf8");
        fs::write(&path, [0xff, 0xfe, 0xfd]).unwrap();

        let err = Manifest::load(&path).unwrap_err();

        assert!(matches!(err, error::TensorFsError::ManifestReadError));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn from_path_returns_error_for_unsorted_segments() {
        let path = temp_file_path("unsorted");
        let mut manifest = sample_manifest();
        manifest.files[0].segments = vec![
            Segment {
                chunk_id: test_chunk_id(2),
                file_offset: 4,
                chunk_offset: 0,
                len: 4,
            },
            Segment {
                chunk_id: test_chunk_id(1),
                file_offset: 0,
                chunk_offset: 0,
                len: 4,
            },
        ];

        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let err = Manifest::load(&path).unwrap_err();

        assert!(matches!(err, error::TensorFsError::ManifestValidationError));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn from_path_returns_error_for_overlapping_segments() {
        let path = temp_file_path("overlap");
        let mut manifest = sample_manifest();
        manifest.files[0].segments = vec![
            Segment {
                chunk_id: test_chunk_id(1),
                file_offset: 0,
                chunk_offset: 0,
                len: 5,
            },
            Segment {
                chunk_id: test_chunk_id(2),
                file_offset: 4,
                chunk_offset: 0,
                len: 4,
            },
        ];

        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let err = Manifest::load(&path).unwrap_err();

        assert!(matches!(err, error::TensorFsError::ManifestValidationError));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn from_path_returns_error_for_segments_out_of_bounds() {
        let path = temp_file_path("out-of-bounds");
        let mut manifest = sample_manifest();
        manifest.files[0].segments[1].len = 5;

        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let err = Manifest::load(&path).unwrap_err();

        assert!(matches!(err, error::TensorFsError::ManifestValidationError));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn from_path_returns_error_for_incorrect_offsets() {
        let path = temp_file_path("incorrect-offsets");
        let mut manifest = sample_manifest();
        manifest.files[0].segments[0].chunk_offset = u64::MAX;

        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let err = Manifest::load(&path).unwrap_err();

        assert!(matches!(err, error::TensorFsError::ManifestValidationError));

        let _ = fs::remove_file(path);
    }
}
