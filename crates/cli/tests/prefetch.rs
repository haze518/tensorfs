use std::fs;
use std::process::Command as StdCommand;
use std::process::Stdio;

use assert_cmd::Command as AssertCommand;
use predicates::str::contains;

#[test]
#[ignore = "downloads a large model from Hugging Face"]
fn prefetch_qwen_real_network() {
    let temp = tempfile::tempdir().unwrap();
    let mut command = StdCommand::new(assert_cmd::cargo::cargo_bin("cli"));

    if let Ok(token) = std::env::var("HF_TOKEN") {
        command.env("HF_TOKEN", token);
    }

    let status = command
        .current_dir(temp.path())
        .env("RUST_LOG", default_rust_log())
        .args(["prefetch", "hf://Qwen/Qwen2.5-0.5B"])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .unwrap();

    assert!(status.success());

    assert!(temp.path().join("manifests/Qwen/Qwen2.5-0.5B").exists());

    let mut cas_entries = fs::read_dir(temp.path().join("cas")).unwrap();
    assert!(cas_entries.next().is_some());
}

#[test]
fn help_lists_prefetch_command() {
    AssertCommand::cargo_bin("cli")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(contains("prefetch"));
}

#[test]
fn prefetch_rejects_non_hf_model_ref_without_network() {
    let temp = tempfile::tempdir().unwrap();

    AssertCommand::cargo_bin("cli")
        .unwrap()
        .current_dir(temp.path())
        .args(["prefetch", "Qwen/Qwen2.5-0.5B"])
        .assert()
        .failure()
        .stderr(contains("Invalid"));
}

#[test]
fn prefetch_rejects_unsafe_hf_model_ref_without_network() {
    let temp = tempfile::tempdir().unwrap();

    AssertCommand::cargo_bin("cli")
        .unwrap()
        .current_dir(temp.path())
        .args(["prefetch", "hf://../outside"])
        .assert()
        .failure()
        .stderr(contains("Invalid"));

    assert!(!temp.path().join("outside").exists());
    assert!(!temp.path().join("manifests/outside").exists());
}

fn default_rust_log() -> String {
    std::env::var("RUST_LOG")
        .unwrap_or_else(|_| "cli=info,api=info,fetch=info,tensorfs=warn,cas=warn".to_string())
}
