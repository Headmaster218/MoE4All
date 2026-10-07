//! Reading a control-vector file: one `[n_embd]` f32 direction per layer.
//!
//! The file is a GGUF whose `general.architecture` is `controlvector` and whose tensors are named
//! `direction.1`, `direction.2`, … — llama.cpp's `--cvec-dir per-layer` layout. `direction.{N}` is
//! the direction of the zero-based layer `N - 1`: the names are one-based, every layer index in
//! this engine is not, and the only thing that could notice the difference is a projection that
//! quietly happens one layer late.

use crate::QWEN4EXP;
use anyhow::{anyhow, Context as _, Result};
use infr_core::loader::TensorInfo;
use infr_core::tensor::DType;
use infr_core::WeightSource;
use infr_gguf::Gguf;
use std::path::Path;

/// `general.architecture` of a control-vector file.
const ARCH: &str = "controlvector";
/// First layer this file describes: `direction.1` is layer 0.
const FIRST_TENSOR_INDEX: usize = 1;

/// Every layer's direction, unit-normalized, as one contiguous `[n_layer_dirs][n_embd]` f32 block.
///
/// Row `l` belongs to the zero-based layer `l`, so a graph builder passes one element offset
/// ([`ControlVectors::dir_off`]) instead of carrying a second index table alongside the buffer.
#[derive(Debug, Clone)]
pub struct ControlVectors {
    n_embd: usize,
    n_layer_dirs: usize,
    data: Vec<f32>,
}

impl ControlVectors {
    /// Width of one direction, i.e. the model width this file was loaded against.
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// How many layers this file covers, i.e. layers `0..n_layer_dirs`.
    pub fn n_layer_dirs(&self) -> usize {
        self.n_layer_dirs
    }

    /// Highest layer index this file can serve (INCLUSIVE). `load` never returns an empty one, so
    /// this cannot underflow.
    pub fn last_layer(&self) -> usize {
        self.n_layer_dirs - 1
    }

    /// Element offset of layer `l`'s direction inside the block. Checked rather than multiplied
    /// blindly: the graph builder asks for it per layer, and a layer this file does not cover is a
    /// bug in the layer range, not something to read past the end of the buffer for.
    pub fn dir_off(&self, l: usize) -> usize {
        debug_assert!(l < self.n_layer_dirs);
        l * self.n_embd
    }

    /// The whole block, for the one `Backend::upload` that carries it.
    pub fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }

    /// Size of that upload, in bytes.
    pub fn byte_len(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

/// Read `path` and prepare its directions for a model of width `n_embd`.
///
/// Every check here is a check on a downloaded file, which is remote input: the architecture, the
/// width, the dtype, and the contiguity of the `direction.N` index all have to be verified rather
/// than trusted, because each one fails LATER as a plausible-looking wrong number in the residual
/// rather than as an error. A direction that is not finite is rejected for the same reason — one
/// `NaN` in one layer's vector would quietly become a `NaN` row in every token that follows.
pub fn load(path: &Path, n_embd: usize) -> Result<ControlVectors> {
    let g = Gguf::open(path).with_context(|| format!("open {}", path.display()))?;
    let md = g.metadata();
    let arch = md.str("general.architecture").unwrap_or("");
    if arch != ARCH {
        return Err(anyhow!(
            "{} is not a control-vector file (general.architecture = `{arch}`, expected \
             `{ARCH}`) — point {} at the direction file, not at a model",
            path.display(),
            crate::config::ENV_VECTOR
        ));
    }
    // `model_hint` says which model this direction set was measured on. A mismatch is not fatal —
    // the projection is still the projection — but a direction from another model is a direction
    // this model never moved along, so the honest expectation is "this changes little or nothing".
    if let Some(hint) = md.str("controlvector.model_hint") {
        if hint != QWEN4EXP {
            tracing::warn!(
                hint,
                "control vector was measured for {hint}, this engine projects it onto {QWEN4EXP}"
            );
        }
    }

    let by_name: std::collections::HashMap<&str, &TensorInfo> =
        g.tensors().iter().map(|t| (t.name.as_str(), t)).collect();
    let declared = md.u64("controlvector.layer_count").map(|v| v as usize);

    let mut data: Vec<f32> = Vec::new();
    let mut layer = 0usize;
    loop {
        let name = format!("direction.{}", layer + FIRST_TENSOR_INDEX);
        // An absent `direction.N` ends the set: the file carries what it carries, and a gap in the
        // middle is caught below rather than silently shortening the range.
        let Some(info) = by_name.get(name.as_str()) else {
            break;
        };
        if info.dtype != DType::F32 {
            return Err(anyhow!(
                "{}: '{name}' is {:?}, but a direction is projected against an f32 residual",
                path.display(),
                info.dtype
            ));
        }
        if info.shape.as_slice() != [n_embd] {
            return Err(anyhow!(
                "{}: '{name}' is {:?}, expected one vector of this model's width ({n_embd}) \
                 — this direction set was measured on a different model",
                path.display(),
                info.shape
            ));
        }
        let bytes = g
            .tensor_bytes(&name)
            .with_context(|| path.display().to_string())?;
        let mut row = Vec::with_capacity(n_embd);
        for c in bytes.chunks_exact(4) {
            row.push(f32::from_le_bytes(c.try_into().unwrap()));
        }
        // Unit-normalize. The projection is `h -= (h·v)v`, which removes exactly the component
        // along `v` when `v` has length 1; the producer ships unit vectors, and normalizing an
        // already-unit vector is a no-op up to rounding, so doing it here makes the file's
        // "unit-normalized per layer" a property of the code rather than a hope.
        let mut norm = 0.0f32;
        for x in row.iter() {
            norm += x * x;
        }
        let norm = norm.sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return Err(anyhow!(
                "{}: '{name}' is not a usable direction (its length is {norm})",
                path.display()
            ));
        }
        for x in row.iter_mut() {
            *x /= norm;
        }
        data.extend_from_slice(&row);
        layer += 1;
    }

    let n_layer_dirs = layer;
    if n_layer_dirs == 0 {
        return Err(anyhow!(
            "{} carries no `direction.{FIRST_TENSOR_INDEX}` — it is not a per-layer control-vector \
             file",
            path.display()
        ));
    }
    if let Some(d) = declared {
        if d != n_layer_dirs {
            return Err(anyhow!(
                "{} declares controlvector.layer_count = {d} but carries {n_layer_dirs} \
                 `direction.*` tensors — the file is truncated or was written by two producers",
                path.display()
            ));
        }
    }
    Ok(ControlVectors {
        n_embd,
        n_layer_dirs,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A GGUF writer small enough to keep next to the reader it is testing: header, the three KV
    /// pairs a control-vector file carries, and one f32 tensor per row of `rows`.
    fn write_controlvector(path: &Path, rows: &[Vec<f32>], width: usize) {
        fn s(out: &mut Vec<u8>, v: &str) {
            out.extend_from_slice(&(v.len() as u64).to_le_bytes());
            out.extend_from_slice(v.as_bytes());
        }
        // A GGUF value tag is a u32, not a byte: writing it as one byte desynchronizes the whole
        // KV section, and the reader's complaint is a confusing "unknown value type".
        fn t(out: &mut Vec<u8>, v: u32) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(rows.len() as u64).to_le_bytes());
        b.extend_from_slice(&3u64.to_le_bytes());
        // Value types are this reader's table: 4 = UINT32, 8 = STRING.
        s(&mut b, "general.architecture");
        t(&mut b, 8);
        s(&mut b, "controlvector");
        s(&mut b, "controlvector.model_hint");
        t(&mut b, 8);
        s(&mut b, "qwen4exp");
        s(&mut b, "controlvector.layer_count");
        t(&mut b, 4);
        b.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        for i in 0..rows.len() {
            s(&mut b, &format!("direction.{}", i + 1));
            b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            b.extend_from_slice(&(width as u64).to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes()); // ggml_type F32
            b.extend_from_slice(&((i * width * 4) as u64).to_le_bytes());
        }
        while b.len() % 32 != 0 {
            b.push(0);
        }
        for row in rows {
            for x in row {
                b.extend_from_slice(&x.to_le_bytes());
            }
        }
        std::fs::write(path, b).unwrap();
    }

    fn fixture(rows: &[Vec<f32>], width: usize) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cvec.gguf");
        write_controlvector(&path, rows, width);
        (dir, path)
    }

    #[test]
    fn loads_rows_and_unit_normalizes_them() {
        // Row 0 is 5× too long, row 1 is already unit; both must come out at length 1.
        let (_d, path) = fixture(&[vec![3.0, 4.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]], 4);
        let cv = load(&path, 4).unwrap();
        assert_eq!(cv.n_embd(), 4);
        assert_eq!(cv.n_layer_dirs(), 2);
        assert_eq!(cv.last_layer(), 1);
        assert_eq!(cv.byte_len(), 8 * std::mem::size_of::<f32>());
        assert_eq!(cv.dir_off(1), 4);
        let r0 = &cv.bytes()[cv.dir_off(0) * 4..cv.dir_off(0) * 4 + 4 * 4];
        let r0: Vec<f32> = bytemuck::cast_slice(r0).to_vec();
        assert!((r0[0] - 0.6).abs() < 1e-6 && (r0[1] - 0.8).abs() < 1e-6);
        let norm1: f32 = cv.bytes()[cv.byte_len() / 2..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!((norm1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_a_width_that_is_not_the_models() {
        let (_d, path) = fixture(&[vec![1.0, 0.0, 0.0, 0.0]], 4);
        let e = load(&path, 8).unwrap_err().to_string();
        assert!(
            e.contains("expected one vector of this model's width (8)"),
            "{e}"
        );
    }

    #[test]
    fn rejects_a_file_that_is_not_a_control_vector() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&("general.architecture".len() as u64).to_le_bytes());
        b.extend_from_slice(b"general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&("qwen4exp".len() as u64).to_le_bytes());
        b.extend_from_slice(b"qwen4exp");
        std::fs::write(&path, b).unwrap();
        let e = load(&path, 4).unwrap_err().to_string();
        assert!(e.contains("not a control-vector file"), "{e}");
    }

    #[test]
    fn rejects_an_all_zero_direction() {
        let (_d, path) = fixture(&[vec![0.0, 0.0, 0.0, 0.0]], 4);
        let e = load(&path, 4).unwrap_err().to_string();
        assert!(e.contains("not a usable direction"), "{e}");
    }
}
