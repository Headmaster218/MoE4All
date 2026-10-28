use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    println!("cargo:rerun-if-env-changed=INFR_GIT_COMMIT");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    for script in ["benchmark-hardware.ps1", "benchmark-hardware.sh"] {
        println!("cargo:rerun-if-changed=../../scripts/{script}");
    }
    for file in ["HEAD", "packed-refs"] {
        if let Some(path) = git(&["rev-parse", "--git-path", file]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git(&["rev-parse", "--git-path", &reference]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(commit) = std::env::var("INFR_GIT_COMMIT")
        .ok()
        .or_else(|| git(&["rev-parse", "HEAD"]))
    {
        if (7..=64).contains(&commit.len()) && commit.bytes().all(|b| b.is_ascii_hexdigit()) {
            println!("cargo:rustc-env=INFR_GIT_COMMIT={commit}");
        }
    }
}
