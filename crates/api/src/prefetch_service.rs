use core::cas::Cas;
use core::error::TensorFsError;
use core::manifest::Manifest;
use core::source::RemoteSource;
use std::path::PathBuf;

use fetch::model_importert::ModelImporter;

pub struct PrefetchService<R, C> {
    importer: ModelImporter<R, C>,
    manifest_dir: PathBuf,
}

impl<R: RemoteSource, C: Cas> PrefetchService<R, C> {
    pub fn new(importer: ModelImporter<R, C>, manifest_dir: PathBuf) -> Self {
        Self {
            importer,
            manifest_dir,
        }
    }

    pub async fn prefetch(&self, model_id: &str) -> Result<Manifest, TensorFsError> {
        let path = manifest_path(&self.manifest_dir, model_id)?;
        tracing::info!(
            model_id = %model_id,
            manifest_path = %path.display(),
            "prefetch service started"
        );

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let on_progress =
            |manifest: &Manifest| -> Result<(), TensorFsError> { manifest.save(&path) };

        let manifest = match Manifest::load(&path) {
            Ok(manifest) => {
                let snapshot = self
                    .importer
                    .snapshot(model_id, Some(&manifest.revision))
                    .await?;
                self.importer.resume(manifest, snapshot, on_progress).await
            }
            Err(TensorFsError::NotFound) => {
                let snapshot = self.importer.snapshot(model_id, None).await?;
                self.importer.download(snapshot, on_progress).await
            }
            Err(err) => return Err(err),
        }?;

        tracing::info!(
            model_id = %model_id,
            manifest_path = %path.display(),
            files = manifest.files.len(),
            "saving manifest"
        );
        manifest.save(&path)?;
        Ok(manifest)
    }
}

fn manifest_path(manifest_dir: &std::path::Path, model_id: &str) -> Result<PathBuf, TensorFsError> {
    let mut path = manifest_dir.to_path_buf();
    let mut has_segment = false;

    for segment in model_id.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." || segment.contains('\\') {
            return Err(TensorFsError::InvalidArgument);
        }

        has_segment = true;
        path.push(segment);
    }

    if !has_segment {
        return Err(TensorFsError::InvalidArgument);
    }

    Ok(path)
}
