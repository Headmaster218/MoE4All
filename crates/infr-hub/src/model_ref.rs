//! Parsed model reference: a HuggingFace `org/repo[:sel]` (no prefix needed) or a filesystem path.
//! `sel` is a quant (`Q4_K_M`, case-insensitive — matches llama.cpp's `-hf`) or an explicit `*.gguf`
//! filename; absent → the default quant. A bare `hf:`/`huggingface:` prefix is accepted but optional.

use infr_core::error::Result;
use std::path::{Path, PathBuf};

/// A parsed model reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelRef {
    /// `org/repo[:sel]` — `sel` is a quant (e.g. `Q4_K_M`) or an explicit `*.gguf` filename.
    Repo { repo: String, sel: Option<String> },
    /// A filesystem path to a `.gguf`.
    Path(PathBuf),
}

impl ModelRef {
    /// Parse a reference. A string that looks like a HuggingFace `org/repo[:sel]` becomes [`Repo`];
    /// anything else (an existing file, a path-like string, or a `*.gguf`) becomes [`Path`].
    ///
    /// - `org/repo`                 → `Repo { repo, sel: None }` (default quant)
    /// - `org/repo:Q4_K_M`          → `Repo { repo, sel: Some("Q4_K_M") }`
    /// - `org/repo:model.gguf`      → `Repo { repo, sel: Some("model.gguf") }`
    /// - `hf:org/repo` (legacy)     → same as `org/repo` (prefix stripped)
    /// - `./m.gguf`, `/abs/m.gguf`  → `Path`
    pub fn parse(s: &str) -> Result<Self> {
        // An existing file is always a path (covers odd names that also look like a repo).
        if Path::new(s).is_file() {
            return Ok(ModelRef::Path(PathBuf::from(s)));
        }
        // Legacy/explicit prefix is accepted but not required.
        let body = s
            .strip_prefix("hf:")
            .or_else(|| s.strip_prefix("huggingface:"))
            .unwrap_or(s);
        let (repo, sel) = match body.split_once(':') {
            Some((r, sel)) if !sel.is_empty() => (r, Some(sel.to_owned())),
            // A lone trailing `:` (`org/repo:`) has an empty selector — strip the colon so it does
            // not leak into the repo name (which would 404 on every resolve URL).
            _ => (body.strip_suffix(':').unwrap_or(body), None),
        };
        if is_repo(repo) {
            Ok(ModelRef::Repo {
                repo: repo.to_owned(),
                sel,
            })
        } else {
            Ok(ModelRef::Path(PathBuf::from(s)))
        }
    }
}

/// Does `repo` look like a HuggingFace `org/repo` (vs a filesystem path)? It must contain a `/`, not
/// be path-rooted (`/ . ~`), and not itself name a `.gguf` file.
fn is_repo(repo: &str) -> bool {
    repo.contains('/')
        && !repo.starts_with(['/', '.', '~'])
        && !repo.to_lowercase().ends_with(".gguf")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn repo(repo: &str, sel: Option<&str>) -> ModelRef {
        ModelRef::Repo {
            repo: repo.into(),
            sel: sel.map(Into::into),
        }
    }

    #[test]
    fn bare_repo_no_sel() {
        assert_eq!(
            ModelRef::parse("unsloth/Qwen3-14B-GGUF").unwrap(),
            repo("unsloth/Qwen3-14B-GGUF", None)
        );
    }

    #[test]
    fn repo_with_quant() {
        assert_eq!(
            ModelRef::parse("unsloth/Qwen3-14B-GGUF:Q4_K_M").unwrap(),
            repo("unsloth/Qwen3-14B-GGUF", Some("Q4_K_M"))
        );
    }

    #[test]
    fn repo_with_explicit_file() {
        assert_eq!(
            ModelRef::parse("org/repo:model-Q4_K_M.gguf").unwrap(),
            repo("org/repo", Some("model-Q4_K_M.gguf"))
        );
    }

    #[test]
    fn legacy_hf_prefix_optional() {
        assert_eq!(
            ModelRef::parse("hf:org/repo:Q4_K_M").unwrap(),
            repo("org/repo", Some("Q4_K_M"))
        );
        assert_eq!(
            ModelRef::parse("huggingface:org/repo").unwrap(),
            repo("org/repo", None)
        );
    }

    #[test]
    fn paths_are_paths() {
        for p in [
            "/abs/m.gguf",
            "./m.gguf",
            "../m.gguf",
            "~/m.gguf",
            "model.gguf",
        ] {
            assert_eq!(
                ModelRef::parse(p).unwrap(),
                ModelRef::Path(PathBuf::from(p))
            );
        }
    }

    #[test]
    fn trailing_colon_stripped() {
        // `org/repo:` — empty selector, the colon must NOT stay in the repo name.
        assert_eq!(
            ModelRef::parse("org/repo:").unwrap(),
            repo("org/repo", None)
        );
        // With the legacy prefix too.
        assert_eq!(
            ModelRef::parse("hf:org/repo:").unwrap(),
            repo("org/repo", None)
        );
    }

    #[test]
    fn relative_gguf_in_subdir_is_path() {
        // ends in .gguf → a file, not a repo, even with a slash
        assert_eq!(
            ModelRef::parse("models/foo.gguf").unwrap(),
            ModelRef::Path(PathBuf::from("models/foo.gguf"))
        );
    }
}
