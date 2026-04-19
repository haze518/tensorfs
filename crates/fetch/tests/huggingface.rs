use fetch::remote_source::{HFClient, RemoteSource};
use reqwest::Url;
use serde_json::Value;

fn make_hf_client() -> HFClient {
    HFClient::new(
        Url::parse("https://huggingface.co/").expect("valid HF base URL"),
        None,
    )
}

#[tokio::test]
#[ignore]
async fn hf_happy_path_lists_files_and_reads_safetensors_header() {
    let client = make_hf_client();

    let files = client
        .list_model_files("Qwen/Qwen2.5-0.5B")
        .await
        .expect("model files should be listed");
    assert!(!files.is_empty(), "expected non-empty file list");

    let safetensors = files
        .iter()
        .find(|file| file.path.ends_with(".safetensors"))
        .expect("expected at least one .safetensors file");

    let tensors = client
        .fetch_safetensors_header(&safetensors.url)
        .await
        .expect("safetensors header should be fetched");
    assert!(!tensors.is_empty(), "expected non-empty tensor metadata");

    assert!(
        tensors.iter().any(|tensor| {
            let value = serde_json::to_value(tensor).expect("TensorMeta should serialize to JSON");
            value
                .get("length")
                .and_then(Value::as_u64)
                .is_some_and(|length| length > 0)
        }),
        "expected at least one tensor with length > 0"
    );
}
