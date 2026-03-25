use std::fs;
use std::path::Path;

fn main() {
    tracing_subscriber::fmt::init();

    println!("Starting tensorfs...");

    if let Err(err) = run() {
        eprintln!("fatal error: {:?}", err);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    init_dirs()?;

    println!("tensorfs started successfully");

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
