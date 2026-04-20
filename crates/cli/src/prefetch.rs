use std::path::PathBuf;
use std::str::FromStr;

use api::prefetch_service::PrefetchService;
use cas::fs::FsCas;
use fetch::hf::HFClient;
use fetch::model_importert::ModelImporter;
use reqwest::Url;

use crate::model_ref::ModelRef;

pub async fn run(source: &str) -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!(source = %source, "starting prefetch");

    let model_ref = ModelRef::from_str(source)?;

    match model_ref {
        ModelRef::HuggingFace { model_id } => {
            let token = std::env::var("HF_TOKEN").ok();
            tracing::info!(
                model_id = %model_id,
                cas_dir = "cas",
                manifest_dir = "manifests",
                has_hf_token = token.is_some(),
                "resolved Hugging Face model"
            );

            let client = HFClient::new(Url::parse("https://huggingface.co/")?, token);
            let storage = FsCas::new(PathBuf::from("cas"));
            let importer = ModelImporter::new(client, storage);
            let service = PrefetchService::new(importer, PathBuf::from("manifests"));

            let manifest = service.prefetch(&model_id).await?;
            tracing::info!(
                model_id = %model_id,
                files = manifest.files.len(),
                "prefetch complete"
            );

            println!(
                "prefetched {}: {} files written",
                model_id,
                manifest.files.len()
            );
        }
    }

    Ok(())
}
