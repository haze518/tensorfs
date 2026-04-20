use crate::error::TensorFsError;
use crate::manifest::{File, Manifest, Segment};
use crate::safetensors::TensorMeta;
use crate::source::RemoteFile;

pub struct ServiceFileLayout {
    pub file: RemoteFile,
    pub tensors: Option<Vec<TensorMeta>>,
}

pub fn build_manifest(
    model_id: &str,
    source_files: Vec<ServiceFileLayout>,
) -> Result<Manifest, TensorFsError> {
    let mut files: Vec<File> = Vec::with_capacity(source_files.len());
    for sl in source_files {
        if let Some(tensors) = sl.tensors {
            if tensors.len() == 0 {
                return Err(TensorFsError::ValidationError);
            }

            let mut segments: Vec<Segment> = Vec::with_capacity(tensors.len() + 1);

            let first_offset = tensors[0].offset;
            if first_offset > 0 {
                segments.push(Segment::new(0, 0, first_offset));
            }

            for tensor in tensors {
                segments.push(Segment::new(tensor.offset, 0, tensor.length));
            }

            files.push(File::new(&sl.file.path, sl.file.size, segments));
        } else {
            let segment = Segment::new(0, 0, sl.file.size);
            files.push(File::new(&sl.file.path, sl.file.size, vec![segment]));
        }
    }

    Manifest::new(model_id, files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::TensorFsError;
    use crate::safetensors::Dtype;
    use std::time::{SystemTime, UNIX_EPOCH};
    use url::Url;

    fn remote_file(path: &str, size: u64) -> RemoteFile {
        RemoteFile {
            path: path.to_string(),
            size,
            url: Url::parse("https://example.com/model").unwrap(),
        }
    }

    fn service_file_layout(
        path: &str,
        size: u64,
        tensors: Option<Vec<TensorMeta>>,
    ) -> ServiceFileLayout {
        ServiceFileLayout {
            file: remote_file(path, size),
            tensors,
        }
    }

    fn tensor(offset: u64, length: u64) -> TensorMeta {
        TensorMeta {
            name: format!("tensor_{offset}"),
            dtype: Dtype::F32,
            shape: vec![length / 4],
            offset,
            length,
        }
    }

    fn temp_file_path(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        std::env::temp_dir().join(format!(
            "tensorfs-manifest-builder-test-{name}-{}-{nanos}.json",
            std::process::id()
        ))
    }

    #[test]
    fn builds_manifest_for_plain_file() {
        let manifest = build_manifest(
            "org/model",
            vec![service_file_layout("plain.bin", 128, None)],
        )
        .unwrap();

        assert_eq!(manifest.files.len(), 1);

        let file = &manifest.files[0];
        assert_eq!(file.path, "plain.bin");
        assert_eq!(file.size, 128);
        assert_eq!(file.segments.len(), 1);
        assert_eq!(file.segments[0].file_offset, 0);
        assert_eq!(file.segments[0].chunk_offset, 0);
        assert_eq!(file.segments[0].len, 128);
    }

    #[test]
    fn builds_manifest_for_safetensors_with_header_segment() {
        let manifest = build_manifest(
            "org/model",
            vec![service_file_layout(
                "model.safetensors",
                200,
                Some(vec![tensor(100, 50), tensor(150, 50)]),
            )],
        )
        .unwrap();

        let file = &manifest.files[0];
        assert_eq!(file.segments.len(), 3);

        assert_eq!(file.segments[0].file_offset, 0);
        assert_eq!(file.segments[0].chunk_offset, 0);
        assert_eq!(file.segments[0].len, 100);

        assert_eq!(file.segments[1].file_offset, 100);
        assert_eq!(file.segments[1].chunk_offset, 0);
        assert_eq!(file.segments[1].len, 50);

        assert_eq!(file.segments[2].file_offset, 150);
        assert_eq!(file.segments[2].chunk_offset, 0);
        assert_eq!(file.segments[2].len, 50);
    }

    #[test]
    fn builds_manifest_for_safetensors_without_header_segment() {
        let manifest = build_manifest(
            "org/model",
            vec![service_file_layout(
                "model.safetensors",
                100,
                Some(vec![tensor(0, 40), tensor(40, 60)]),
            )],
        )
        .unwrap();

        let file = &manifest.files[0];
        assert_eq!(file.segments.len(), 2);

        assert_eq!(file.segments[0].file_offset, 0);
        assert_eq!(file.segments[0].chunk_offset, 0);
        assert_eq!(file.segments[0].len, 40);

        assert_eq!(file.segments[1].file_offset, 40);
        assert_eq!(file.segments[1].chunk_offset, 0);
        assert_eq!(file.segments[1].len, 60);
    }

    #[test]
    fn returns_error_for_empty_tensors() {
        let err = build_manifest(
            "org/model",
            vec![service_file_layout("empty.safetensors", 10, Some(vec![]))],
        )
        .unwrap_err();

        assert!(matches!(err, TensorFsError::ValidationError));
    }

    #[test]
    fn returns_error_for_gap_between_tensors() {
        let err = build_manifest(
            "org/model",
            vec![service_file_layout(
                "gapped.safetensors",
                250,
                Some(vec![tensor(100, 50), tensor(200, 50)]),
            )],
        )
        .unwrap_err();

        assert!(matches!(err, TensorFsError::ManifestValidationError));
    }

    #[test]
    fn saves_and_loads_built_manifest() {
        let path = temp_file_path("round-trip");
        let manifest = build_manifest(
            "org/model",
            vec![service_file_layout(
                "model.safetensors",
                200,
                Some(vec![tensor(100, 50), tensor(150, 50)]),
            )],
        )
        .unwrap();

        manifest.save(&path).unwrap();

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

        for (actual, expected) in loaded.files[0]
            .segments
            .iter()
            .zip(manifest.files[0].segments.iter())
        {
            assert_eq!(actual.file_offset, expected.file_offset);
            assert_eq!(actual.chunk_offset, expected.chunk_offset);
            assert_eq!(actual.len, expected.len);
        }

        let _ = std::fs::remove_file(path);
    }
}
