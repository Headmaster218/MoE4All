//! Local raw results for human-format benchmarks. JSON callers already receive the same data
//! on stdout; this module never changes that stream or starts an upload.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_RESULT: AtomicU64 = AtomicU64::new(0);

fn default_root() -> PathBuf {
    if let Some(root) = std::env::var_os("MOE4ALL_WORK_ROOT").filter(|root| !root.is_empty()) {
        PathBuf::from(root).join("results/moe4all/manual-benchmark")
    } else {
        PathBuf::from("benchmark-results")
    }
}

pub fn save(raw: &str) -> std::io::Result<PathBuf> {
    save_at(&default_root(), raw)
}

fn save_at(root: &Path, raw: &str) -> std::io::Result<PathBuf> {
    use std::fs::OpenOptions;

    // These are local raw artifacts, not submission IDs. Never replace an existing result,
    // even with a coarse/repeated system clock or another process sharing the directory.
    std::fs::create_dir_all(root)?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let (path, mut file) = loop {
        let ordinal = NEXT_RESULT.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            "bench-{stamp}-{}-{ordinal}.json",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => break (path, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    if let Err(error) = file
        .write_all(raw.as_bytes())
        .and_then(|()| file.write_all(b"\n"))
        .and_then(|()| file.flush())
    {
        drop(file);
        let _ = std::fs::remove_file(&path);
        return Err(error);
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_results_are_unique_and_preserve_all_collected_metadata() {
        let root =
            std::env::temp_dir().join(format!("moe4all-benchmark-export-{}", std::process::id()));
        let raw = r#"[{"avg_ts":123.4,"benchmark_metadata":{"hardware":{"ram":{"total_bytes":68719476736},"gpu":{"pcie_current_width":8},"model_storage":{"read_probe":{"sequential":{"bytes":1048576}}}}}}]"#;
        let first = save_at(&root, raw).unwrap();
        let second = save_at(&root, raw).unwrap();
        assert_ne!(first, second);
        assert_eq!(std::fs::read_to_string(&first).unwrap(), format!("{raw}\n"));
        assert_eq!(
            std::fs::read_to_string(&second).unwrap(),
            format!("{raw}\n")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unwritable_destination_does_not_replace_existing_data() {
        let root = std::env::temp_dir().join(format!(
            "moe4all-benchmark-export-blocked-{}",
            std::process::id()
        ));
        std::fs::write(&root, "existing state").unwrap();
        assert!(save_at(&root, "[]").is_err());
        assert_eq!(std::fs::read_to_string(&root).unwrap(), "existing state");
        std::fs::remove_file(root).unwrap();
    }
}
