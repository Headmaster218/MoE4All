//! **The B9 guard**: no bare `print!`/`println!`/`eprint!`/`eprintln!` in production code.
//!
//! Diagnostics belong on `tracing` — a library must never write to the process's streams
//! directly, and a bare print carries no level, no fields and no filter, so `RUST_LOG` cannot
//! turn it up or down and `infr serve`'s request log has nothing to interleave it with.
//!
//! Four categories are sanctioned, and this test knows all four:
//!
//! 1. **In-crate `#[cfg(test)]` modules.** Each file is cut at its FIRST `#[cfg(test)]` and only
//!    the half above it is scanned — that is test output, and `cargo test` captures it.
//! 2. **`build.rs`.** `cargo:rerun-if-changed` is a build protocol, not logging. Not scanned.
//! 3. **Program OUTPUT**, marked at the site with a `// print-ok: <reason>` comment: generated
//!    tokens, `infr bench` / `infr compare` tables, `--json`, `infr devices`, the serve banner.
//!    This is the CLI's contract — users pipe it, and a filterable logger would break it.
//! 4. **Profiling REPORTS**, same marker, but only in [`REPORT_FILES`]: column-aligned tables the
//!    user asked for by setting `INFR_PROFILE`/`INFR_PROF_OPS`, which a per-line `tracing` prefix
//!    would destroy, and which partly run from a C `atexit` hook where no subscriber is left.
//!
//! The marker is deliberately per-SITE. A library crate cannot exempt itself with it: outside
//! [`REPORT_FILES`] and [`OUTPUT_CRATES`] the marker is ignored and the print still fails the
//! test. Adding a file to either list is a visible edit to this test, which is the point.

use std::path::{Path, PathBuf};

/// Crates whose job IS writing to the process's streams: the binaries the user invokes. A
/// `// print-ok:` marker is honoured anywhere in these; everywhere else it is not.
const OUTPUT_CRATES: &[&str] = &["infr-cli"];

/// Library files allowed to carry `// print-ok:` markers, each for one named reason. Exact paths,
/// crate-relative — never a whole crate.
const REPORT_FILES: &[&str] = &[
    // The `INFR_PROFILE` / `INFR_PROF_OPS` exit report. Runs from a C `atexit` hook.
    "infr-prof-rt/src/lib.rs",
    // `OpProf::flush`'s per-op table, which feeds the report above.
    "infr-core/src/prof.rs",
    // infr-metal's `prof.stages` table and its pipeline-cache summary line.
    "infr-metal/src/profile.rs",
    "infr-metal/src/shaders.rs",
    // The shared TEST harness: every caller of `SweepReport::assert_ok` is a `#[test]`.
    "infr-testkit/src/lib.rs",
];

/// The four macros. `write!`/`writeln!` to an explicit stream are fine — they name their sink.
const MACROS: &[&str] = &["println!", "eprintln!", "print!", "eprint!"];

const MARKER: &str = "print-ok:";

#[test]
fn no_bare_print_outside_the_sanctioned_categories() {
    let Some(crates) = repo_crates_dir() else {
        return; // packaged/vendored build — no `crates/` to scan
    };
    let mut offenders: Vec<String> = Vec::new();
    let mut marked = 0usize;
    for entry in std::fs::read_dir(&crates).expect("read crates/") {
        let dir = entry.expect("crates/ entry").path();
        let Some(crate_name) = dir.file_name().and_then(|n| n.to_str()).map(str::to_string) else {
            continue;
        };
        scan_dir(
            &dir.join("src"),
            &crate_name,
            &crates,
            &mut offenders,
            &mut marked,
        );
    }
    offenders.sort_unstable();

    // The scan must have RUN. A refactor that renames `crates/`, moves the sources, or breaks the
    // marker convention would otherwise turn this test into a green light wired to nothing.
    assert!(
        marked >= 40,
        "the scanner found only {marked} `// {MARKER}` markers — it expects the sanctioned CLI \
         output and profile-report sites (54 in infr-cli plus the report tables). Either the \
         marker convention changed or the scan matched nothing; fix the scanner, do not lower \
         this floor."
    );

    assert!(
        offenders.is_empty(),
        "these production sites print straight to the process's streams. Route the diagnostic \
         through `tracing` (`warn!` for a degradation, `info!` for lifecycle, `debug!`/`trace!` \
         for anything per-layer or per-token), or — if it is genuinely program OUTPUT — mark the \
         site with a `// {MARKER} <reason>` comment on the line above. See this test's module doc \
         for the four sanctioned categories:\n{}",
        offenders.join("\n")
    );
}

/// `<repo>/crates`, or `None` when we are not building from the repo.
fn repo_crates_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")) // …/crates/infr-core
        .parent()?
        .to_path_buf();
    dir.is_dir().then_some(dir)
}

fn scan_dir(dir: &Path, crate_name: &str, root: &Path, out: &mut Vec<String>, marked: &mut usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_dir(&path, crate_name, root, out, marked);
        } else if path.extension().is_some_and(|e| e == "rs") {
            scan_file(&path, crate_name, root, out, marked);
        }
    }
}

fn scan_file(
    path: &Path,
    crate_name: &str,
    root: &Path,
    out: &mut Vec<String>,
    marked: &mut usize,
) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    let rel = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    // Category 3/4: whether a `// print-ok:` marker is honoured HERE at all.
    let marker_honoured = OUTPUT_CRATES.contains(&crate_name) || REPORT_FILES.contains(&&*rel);

    let lines: Vec<&str> = text.lines().collect();
    // Category 1: cut at the first `#[cfg(test)]`; everything below it is test output.
    //
    // `starts_with`, not `contains`: B9's own count used `contains` and so cut
    // `infr-vulkan/src/lib.rs` at line 1262 — a DOC COMMENT quoting `#[cfg(test)]` — hiding 12
    // real production `eprintln!`s below it and under-reporting that crate as 11 sites. Only an
    // attribute in column 0 of its own line ends production code. `#[cfg(any(test, …))]` and
    // `#[cfg(any(target_os = "macos", test))]` deliberately do NOT cut: both compile in
    // non-test builds.
    let end = lines
        .iter()
        .position(|l| l.trim_start().starts_with("#[cfg(test)]"))
        .unwrap_or(lines.len());

    for (i, line) in lines[..end].iter().enumerate() {
        let code = line.trim_start();
        // A doc comment or prose naming a macro is not a call.
        if code.starts_with("//") {
            continue;
        }
        let Some(mac) = MACROS.iter().find(|m| contains_macro_call(code, m)) else {
            continue;
        };
        // The marker may sit on the call's own line, or anywhere in the contiguous `//` comment
        // block directly above it (the reasons run to two or three lines).
        let has_marker = code.contains(MARKER) || {
            let mut j = i;
            let mut found = false;
            while j > 0 {
                let above = lines[j - 1].trim_start();
                if !above.starts_with("//") {
                    break;
                }
                found |= above.contains(MARKER);
                j -= 1;
            }
            found
        };
        if has_marker {
            if marker_honoured {
                *marked += 1;
                continue;
            }
            out.push(format!(
                "{}:{}: `{mac}` carries a `{MARKER}` marker, but {crate_name} is a LIBRARY — the \
                 marker is only honoured in {OUTPUT_CRATES:?} and in this test's REPORT_FILES",
                rel,
                i + 1
            ));
            continue;
        }
        out.push(format!("{}:{}: bare `{mac}`", rel, i + 1));
    }
}

/// `mac` appears as a macro CALL, not as the tail of a longer name (`println!` inside `eprintln!`,
/// or a hypothetical `my_print!`). The char before must not be an identifier char.
fn contains_macro_call(code: &str, mac: &str) -> bool {
    let bytes = code.as_bytes();
    let mut from = 0;
    while let Some(rel) = code[from..].find(mac) {
        let at = from + rel;
        let ok = at == 0 || {
            let p = bytes[at - 1];
            !(p.is_ascii_alphanumeric() || p == b'_')
        };
        if ok {
            return true;
        }
        from = at + 1;
    }
    false
}
