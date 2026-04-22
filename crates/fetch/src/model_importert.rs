use std::collections::{HashMap, HashSet};

use tensorfs::cas::Cas;
use tensorfs::chunk::ChunkId;
use tensorfs::error::TensorFsError;
use tensorfs::manifest::Manifest;
use tensorfs::manifest_builder::{ServiceFileLayout, build_manifest};
use tensorfs::source::{RemoteSnapshot, RemoteSource};
use tracing::info;

pub struct ModelImporter<R, C> {
    client: R,
    storage: C,
}

impl<R: RemoteSource, C: Cas> ModelImporter<R, C> {
    pub fn new(client: R, storage: C) -> Self {
        Self { client, storage }
    }

    pub async fn snapshot(
        &self,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<RemoteSnapshot, TensorFsError> {
        self.client.get_snapshot(model_id, revision).await
    }

    pub async fn download<F>(
        &self,
        current_snapshot: RemoteSnapshot,
        mut on_progress: F,
    ) -> Result<Manifest, TensorFsError>
    where
        F: FnMut(&Manifest) -> Result<(), TensorFsError>,
    {
        let mut sfl = Vec::new();
        let mut file_url = HashMap::new();

        for f in current_snapshot.files {
            let mut tensors = None;
            if f.path.ends_with(".safetensors") {
                tensors = Some(self.client.fetch_safetensors_header(&f.url).await?);
            }
            file_url.insert(f.path.clone(), f.url.clone());
            sfl.push(ServiceFileLayout {
                file: f,
                tensors: tensors,
            });
        }

        let mut manifest = build_manifest(&current_snapshot.id, &current_snapshot.revision, sfl)?;

        let total_segments: usize = manifest.files.iter().map(|f| f.segments.len()).sum();
        let mut done = 0;

        info!(model_id = %current_snapshot.id, revision = %current_snapshot.revision, total_segments, "start downloading model");

        for file_idx in 0..manifest.files.len() {
            let file_path = &manifest.files[file_idx].path;
            let url = file_url
                .get(file_path)
                .ok_or(TensorFsError::ManifestValidationError)?;

            for segment_idx in 0..manifest.files[file_idx].segments.len() {
                let segment = &manifest.files[file_idx].segments[segment_idx];
                let bytes = self
                    .client
                    .fetch_range(&url, segment.file_offset, segment.len)
                    .await?;
                let chunk_id = self.storage.put(bytes).await?;

                {
                    let segment = &mut manifest.files[file_idx].segments[segment_idx];
                    segment.chunk_id = chunk_id;
                }

                on_progress(&manifest)?;
                done += 1;

                info!(
                    progress = format!("{}/{}", done, total_segments),
                    "downloading segment"
                );
            }
        }

        info!("download complete");

        Ok(manifest)
    }

    pub async fn resume<F>(
        &self,
        mut manifest: Manifest,
        current_snapshot: RemoteSnapshot,
        mut on_progress: F,
    ) -> Result<Manifest, TensorFsError>
    where
        F: FnMut(&Manifest) -> Result<(), TensorFsError>,
    {
        if manifest.revision != current_snapshot.revision {
            return Err(TensorFsError::ValidationError);
        }
        let mut file_url = HashMap::new();

        for f in current_snapshot.files {
            file_url.insert(f.path.clone(), f.url.clone());
        }

        let total_segments: usize = manifest.files.iter().map(|f| f.segments.len()).sum();

        let mut presented_chunks = HashSet::new();
        let mut done = 0;
        for file in &manifest.files {
            for segment in &file.segments {
                if self.segment_is_present(segment.chunk_id).await? {
                    presented_chunks.insert(segment.chunk_id);
                    done += 1;
                }
            }
        }

        info!(model_id = %current_snapshot.id, revision = %current_snapshot.revision, total_segments, "continue downloading model");
        for file_idx in 0..manifest.files.len() {
            let file_path = &manifest.files[file_idx].path;
            let url = file_url
                .get(file_path)
                .ok_or(TensorFsError::ManifestValidationError)?;

            for segment_idx in 0..manifest.files[file_idx].segments.len() {
                let segment = &manifest.files[file_idx].segments[segment_idx];
                if presented_chunks.contains(&segment.chunk_id) {
                    continue;
                }

                let bytes = self
                    .client
                    .fetch_range(&url, segment.file_offset, segment.len)
                    .await?;

                let chunk_id = self.storage.put(bytes).await?;

                {
                    let segment = &mut manifest.files[file_idx].segments[segment_idx];
                    segment.chunk_id = chunk_id;
                }

                on_progress(&manifest)?;
                done += 1;

                info!(
                    progress = format!("{}/{}", done, total_segments),
                    "downloading segment"
                );
            }
        }

        info!("download complete");
        Ok(manifest)
    }

    async fn segment_is_present(&self, chunk_id: ChunkId) -> Result<bool, TensorFsError> {
        if chunk_id.is_empty() {
            return Ok(false);
        }

        self.storage.exists(chunk_id).await
    }
}
