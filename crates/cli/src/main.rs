use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};

mod model_ref;
mod prefetch;

#[derive(Parser)]
#[command(name = "tensorfs")]
#[command(about = "TensorFS command line interface")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Prefetch { model_id: String },
}

fn main() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "cli=info,api=info,fetch=info,tensorfs=warn,cas=warn".into());
    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");

    if let Err(err) = runtime.block_on(run()) {
        eprintln!("fatal error: {:?}", err);
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    init_dirs()?;

    match cli.command {
        Some(Commands::Prefetch { model_id }) => prefetch::run(&model_id).await?,
        None => println!("tensorfs started successfully"),
    }

    Ok(())
}

fn init_dirs() -> Result<(), Box<dyn std::error::Error>> {
    let dirs = ["data", "manifests", "cas"];

    for dir in dirs {
        if !Path::new(dir).exists() {
            fs::create_dir_all(dir)?;
        }
    }

    Ok(())
}
