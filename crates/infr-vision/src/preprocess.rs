//! Image → ViT input preprocessing for `qwen3vl_merger` mmproj files.
//!
//! Mirrors llama.cpp `tools/mtmd/clip.cpp` + `tools/mtmd/models/qwen3vl.cpp`:
//! smart-resize to multiples of `patch*merge` (32), per-channel normalize, patchify into
//! 16×16×3 patches, then reorder into the spatial-merge-major layout the merger expects.
//! Position embeddings are bilinearly resized from the base grid (`GGML_SCALE_MODE_BILINEAR |
//! GGML_SCALE_FLAG_ALIGN_CORNERS` semantics) and reordered identically, so `patches[i]` and
//! `pos_embed[i]` always describe the same patch.

use crate::ClipConfig;
use anyhow::{bail, Context, Result};
use base64::Engine;
use image::{imageops::FilterType, GenericImageView, ImageReader};
use std::io::Cursor;

/// Hard cap on the number of 16px patches per image (llama.cpp mtmd caps token budget per
/// image similarly; 4096 patches = 1024 merged tokens ≈ 1M pixels, qwen3vl's default budget).
pub const MAX_PATCHES: usize = 4096;

/// One image prepared for the ViT forward.
pub struct PreparedImage {
    /// `[n_patches, 3*patch²]` normalized pixels in merge-major order (see the module docs and
    /// `merge_major_to_patch` for the ordering contract).
    pub patches: Vec<f32>,
    /// Merged grid width in tokens, `img_w / (patch*merge)`.
    pub grid_nx: usize,
    /// Merged grid height in tokens, `img_h / (patch*merge)`.
    pub grid_ny: usize,
    /// `[n_patches, embedding_length]` position embeddings bilinear-resized from the base grid,
    /// in the SAME merge-major order as `patches`.
    pub pos_embed: Vec<f32>,
}

impl PreparedImage {
    /// Total patches (= `grid_nx * grid_ny * merge²`).
    pub fn n_patches(&self) -> usize {
        self.patches.len() / (3 * 16 * 16)
    }
}

/// Decode an OpenAI-style image input into raw image bytes.
///
/// - `data:[<mime>][;base64],<payload>` — base64-decoded.
/// - a bare base64 string — decoded.
/// - `http(s)://…` — **not supported in this stage** (V5 wires fetching); bails with a clear error.
pub fn decode_image_input(url_or_data: &str) -> Result<Vec<u8>> {
    if url_or_data.starts_with("http://") || url_or_data.starts_with("https://") {
        bail!(
            "infr-vision V1 does not fetch http(s) image URLs; pass a data: URI or base64 \
             payload instead"
        );
    }
    let payload = match url_or_data.strip_prefix("data:") {
        Some(rest) => {
            let comma = rest
                .find(',')
                .context("malformed data: URI (missing ',' separator)")?;
            if !rest[..comma].ends_with(";base64") {
                bail!(
                    "data: URI must be base64-encoded (got {:?})",
                    &rest[..comma]
                );
            }
            &rest[comma + 1..]
        }
        None => url_or_data,
    };
    base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .context("image input is not valid base64")
}

/// Smart-resize target dims (llama.cpp mtmd qwen3vl): round each side to the nearest multiple of
/// `factor` (at least one `factor`), then — if the resulting patch count exceeds
/// [`MAX_PATCHES`] — scale the ORIGINAL aspect down by `sqrt(pixels / budget)` and floor each
/// side to a multiple of `factor` (Qwen2/3-VL `smart_resize`'s downscale branch).
pub fn smart_resize(w: u32, h: u32, factor: u32, patch_size: usize) -> (u32, u32) {
    let f = factor.max(1) as f64;
    let round = |v: u32| ((v as f64 / f).round().max(1.0) as u32) * factor.max(1);
    let (mut rw, mut rh) = (round(w), round(h));
    let ps = patch_size.max(1) as u32;
    if (rw / ps) as usize * (rh / ps) as usize > MAX_PATCHES {
        let beta =
            ((w as f64 * h as f64) / (MAX_PATCHES * ps as usize * ps as usize) as f64).sqrt();
        let floor = |v: u32| ((v as f64 / beta / f).floor().max(1.0) as u32) * factor.max(1);
        rw = floor(w);
        rh = floor(h);
    }
    (rw, rh)
}

/// Merge-major sequence index → patch-grid coordinate, for a `nx`×`ny` patch grid with 2×2
/// spatial merge. Blocks are row-major over the merged grid; inside a block the 4 patches are
/// row-major `(dy, dx)`.
///
/// This is the ordering HF's Qwen2/3-VL processor produces
/// (`rearrange("... (h m) (w n) c -> ... (h w) (m n c)")` with m,n = merge size) and the one
/// llama.cpp's reshape/permute of the `[embd, nx, ny]` conv output implements; both patches and
/// position embeddings here use it, which is what the merger's `view(-1, embd*4)` requires.
fn merge_major_to_patch(i: usize, nx: usize, merge: usize) -> (usize, usize) {
    let mx = nx / merge;
    let block = i / (merge * merge);
    let inner = i % (merge * merge);
    let by = block / mx;
    let bx = block % mx;
    let dy = inner / merge;
    let dx = inner % merge;
    (by * merge + dy, bx * merge + dx)
}

/// Public alias over [`merge_major_to_patch`] for the ViT forward's 2D-RoPE position
/// table: merge-major sequence index → `(y, x)` on the PRE-merge patch grid.
pub fn merge_major_pos(i: usize, nx: usize, merge: usize) -> (usize, usize) {
    merge_major_to_patch(i, nx, merge)
}

/// Bilinear resize of the `[embd, base²]` position table to an `out_x`×`out_y` patch grid with
/// ALIGN_CORNERS semantics (ggml `GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS`):
/// output index `i` maps to source coordinate `i * (in-1) / (out-1)`, corners land exactly.
///
/// `table` layout is GGUF ne0-fastest: element `(d, x, y)` at `table[(y*base + x) * embd + d]`.
/// Output is row-major patches: `out[(oy*out_x + ox) * embd + d]`.
pub(crate) fn bilinear_resize_pos_table(
    table: &[f32],
    embd: usize,
    base: usize,
    out_x: usize,
    out_y: usize,
) -> Result<Vec<f32>> {
    if table.len() != base * base * embd {
        bail!(
            "position table has {} elements; expected {} (embd {embd} × base² {base}²)",
            table.len(),
            base * base * embd
        );
    }
    let at = |x: usize, y: usize, d: usize| table[(y * base + x) * embd + d];
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let coord = |i: usize, out: usize| -> (usize, usize, f32) {
        if out <= 1 {
            return (0, 0, 0.0);
        }
        let s = i as f64 * (base - 1) as f64 / (out - 1) as f64;
        let lo = (s.floor() as usize).min(base - 1);
        let hi = (lo + 1).min(base - 1);
        (lo, hi, (s - lo as f64) as f32)
    };
    let mut out = vec![0.0f32; out_x * out_y * embd];
    for oy in 0..out_y {
        let (y0, y1, ty) = coord(oy, out_y);
        for ox in 0..out_x {
            let (x0, x1, tx) = coord(ox, out_x);
            let dst = &mut out[(oy * out_x + ox) * embd..(oy * out_x + ox + 1) * embd];
            for (d, slot) in dst.iter_mut().enumerate() {
                let top = lerp(at(x0, y0, d), at(x1, y0, d), tx);
                let bot = lerp(at(x0, y1, d), at(x1, y1, d), tx);
                *slot = lerp(top, bot, ty);
            }
        }
    }
    Ok(out)
}

/// Decode, resize, normalize, patchify, and reorder one image into ViT input.
///
/// `pos_table_f32` is the dequantized `v.position_embd.weight`, `[embd, base²]` ne0-fastest.
pub fn prepare_image_bytes(
    bytes: &[u8],
    cfg: &ClipConfig,
    pos_table_f32: &[f32],
) -> Result<PreparedImage> {
    let img = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("guessing image format")?
        .decode()
        .context("decoding image")?;
    let (w, h) = img.dimensions();
    let (rw, rh) = smart_resize(w, h, cfg.merge_factor() as u32, cfg.patch_size);
    tracing::debug!(from = ?(w, h), to = ?(rw, rh), "vision smart-resize");
    let rgb = img.resize_exact(rw, rh, FilterType::Triangle).to_rgb8();

    let nx = rw as usize / cfg.patch_size; // patch grid
    let ny = rh as usize / cfg.patch_size;
    let ps = cfg.patch_size;
    let merge = cfg.spatial_merge_size;
    let n_patches = nx * ny;
    let patch_len = 3 * ps * ps;

    // Row-major patchification, channel-planar within a patch: index (dx, dy, c) →
    // dx + ps*(dy + ps*c). This matches GGUF `v.patch_embd.weight` ne0..2 = (kw, kh, c): the
    // conv's im2col flattening is x-fastest, channel-slowest (same as HF's
    // `view(-1, c, t, ps, ps)` channel-planar layout).
    let mut row_major = vec![0.0f32; n_patches * patch_len];
    for py in 0..ny {
        for px in 0..nx {
            let dst = &mut row_major[(py * nx + px) * patch_len..(py * nx + px + 1) * patch_len];
            for c in 0..3usize {
                let mean = cfg.image_mean[c];
                let std = cfg.image_std[c].max(f32::EPSILON);
                for dy in 0..ps {
                    for dx in 0..ps {
                        let v = rgb.get_pixel((px * ps + dx) as u32, (py * ps + dy) as u32)[c];
                        dst[dx + ps * (dy + ps * c)] = (v as f32 / 255.0 - mean) / std;
                    }
                }
            }
        }
    }

    // Position embeddings resized to the patch grid, row-major, then reordered with the SAME
    // merge-major permutation as the patches.
    let pos_row_major =
        bilinear_resize_pos_table(pos_table_f32, cfg.embedding_length, cfg.base_grid, nx, ny)?;

    // Merge-major reorder (see `merge_major_to_patch`).
    let mut patches = vec![0.0f32; n_patches * patch_len];
    let mut pos_embed = vec![0.0f32; n_patches * cfg.embedding_length];
    for i in 0..n_patches {
        let (py, px) = merge_major_to_patch(i, nx, merge);
        let src = py * nx + px;
        patches[i * patch_len..(i + 1) * patch_len]
            .copy_from_slice(&row_major[src * patch_len..(src + 1) * patch_len]);
        let e = cfg.embedding_length;
        pos_embed[i * e..(i + 1) * e].copy_from_slice(&pos_row_major[src * e..(src + 1) * e]);
    }

    Ok(PreparedImage {
        patches,
        grid_nx: nx / merge,
        grid_ny: ny / merge,
        pos_embed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn test_cfg() -> ClipConfig {
        ClipConfig {
            block_count: 2,
            embedding_length: 8,
            feed_forward_length: 16,
            head_count: 2,
            head_dim: 4,
            image_size: 768,
            patch_size: 16,
            spatial_merge_size: 2,
            projection_dim: 4,
            use_gelu: true,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            layer_norm_epsilon: 1e-6,
            is_deepstack_layers: vec![false; 2],
            base_grid: 2,
        }
    }

    /// Encode a solid/procedural RGB image as PNG bytes in memory.
    fn png_of(img: &RgbImage) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png)
            .expect("encode png");
        buf.into_inner()
    }

    // ── smart_resize ────────────────────────────────────────────────────────

    #[test]
    fn smart_resize_rounds_to_merge_factor() {
        // 100 → 96 (3×32), 65 → 64 (2×32).
        assert_eq!(smart_resize(100, 65, 32, 16), (96, 64));
        // Below one factor clamps up to it.
        assert_eq!(smart_resize(10, 10, 32, 16), (32, 32));
        // Already-aligned dims are untouched.
        assert_eq!(smart_resize(768, 512, 32, 16), (768, 512));
    }

    #[test]
    fn smart_resize_caps_patch_count() {
        let (w, h) = smart_resize(8192, 8192, 32, 16);
        assert!((w as usize / 16) * (h as usize / 16) <= MAX_PATCHES);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
        // Aspect preserved within rounding: square stays square.
        assert_eq!(w, h);
    }

    // ── decode_image_input ──────────────────────────────────────────────────

    #[test]
    fn decode_accepts_data_uri_and_bare_base64() {
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"hello");
        assert_eq!(decode_image_input(&b64).expect("bare"), b"hello");
        let uri = format!("data:image/png;base64,{b64}");
        assert_eq!(decode_image_input(&uri).expect("data uri"), b"hello");
    }

    #[test]
    fn decode_rejects_http_and_non_base64_data_uri() {
        assert!(decode_image_input("https://example.com/x.png").is_err());
        assert!(decode_image_input("data:image/png,raw").is_err());
        assert!(decode_image_input("not base64 !!!").is_err());
    }

    // ── bilinear (ALIGN_CORNERS) ────────────────────────────────────────────

    #[test]
    fn bilinear_2x2_to_4x4_align_corners() {
        // base=2, embd=1: values [[0, 10], [20, 30]] at corners (x,y) ∈ {0,1}².
        let table = [0.0f32, 10.0, 20.0, 30.0]; // (x,y): (0,0),(1,0),(0,1),(1,1)
        let out = bilinear_resize_pos_table(&table, 1, 2, 4, 4).expect("resize");
        // Corners map exactly (align-corners): src = i*(1/3).
        assert_eq!(out[0 * 4 + 0], 0.0);
        assert_eq!(out[0 * 4 + 3], 10.0);
        assert_eq!(out[3 * 4 + 0], 20.0);
        assert_eq!(out[3 * 4 + 3], 30.0);
        // (ox=1, oy=0): sx = 1/3 → 0 + (10-0)/3.
        assert!((out[0 * 4 + 1] - 10.0 / 3.0).abs() < 1e-6);
        // (ox=1, oy=1): sx=sy=1/3 → bilinear = (0*4 + 10*2 + 20*2 + 30*1)/9 = 90/9 = 10.
        assert!((out[1 * 4 + 1] - 10.0).abs() < 1e-6);
        // (ox=2, oy=2): sx=sy=2/3 → top = 20/3, bot = 80/3 → 20/3 + (60/3)(2/3) = 20.
        assert!((out[2 * 4 + 2] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn bilinear_identity_when_grids_match() {
        let table: Vec<f32> = (0..16).map(|v| v as f32).collect(); // base=4, embd=1
        let out = bilinear_resize_pos_table(&table, 1, 4, 4, 4).expect("resize");
        assert_eq!(out, table);
    }

    // ── merge-major ordering ────────────────────────────────────────────────

    #[test]
    fn merge_major_index_mapping() {
        // 4x4 patch grid (nx=4), merge=2 → merged grid 2x2, 16 patches.
        // seq 0..4 must be the top-left 2x2 block: (0,0),(0,1),(1,0),(1,1) as (py,px).
        let got: Vec<(usize, usize)> = (0..4).map(|i| merge_major_to_patch(i, 4, 2)).collect();
        assert_eq!(got, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
        // seq 4..8: block (by=0, bx=1) → patches (0,2),(0,3),(1,2),(1,3).
        let got: Vec<(usize, usize)> = (4..8).map(|i| merge_major_to_patch(i, 4, 2)).collect();
        assert_eq!(got, vec![(0, 2), (0, 3), (1, 2), (1, 3)]);
        // The mapping is a permutation of the full grid.
        let mut all: Vec<(usize, usize)> = (0..16).map(|i| merge_major_to_patch(i, 4, 2)).collect();
        all.sort();
        let mut want: Vec<(usize, usize)> =
            (0..4).flat_map(|y| (0..4).map(move |x| (y, x))).collect();
        want.sort();
        assert_eq!(all, want);
    }

    // ── patchify + reorder end-to-end on a 4x4-patch toy image ──────────────

    /// 64×64 image (4×4 patches). Every pixel in patch (px,py) is a solid color whose red byte
    /// encodes the patch: `r = py*4 + px + 1` (so normalized value identifies the patch).
    #[test]
    fn patchify_merge_major_order_and_values() {
        let ps = 16usize;
        let img = RgbImage::from_fn(64, 64, |x, y| {
            let px = x as usize / ps;
            let py = y as usize / ps;
            Rgb([(py * 4 + px + 1) as u8, 0, 0])
        });
        let cfg = test_cfg(); // base_grid=2, embd=8
        let pos_table: Vec<f32> = (0..cfg.base_grid * cfg.base_grid * cfg.embedding_length)
            .map(|v| v as f32)
            .collect();
        let prep = prepare_image_bytes(&png_of(&img), &cfg, &pos_table).expect("prepare");

        assert_eq!((prep.grid_nx, prep.grid_ny), (2, 2));
        assert_eq!(prep.n_patches(), 16);
        assert_eq!(prep.pos_embed.len(), 16 * 8);

        // With mean=std=0.5: normalized = (v/255 - 0.5)/0.5. First element of each patch is the
        // red channel's top-left pixel (index (dx=0,dy=0,c=0) → 0), so it names the source patch.
        let norm = |v: u8| (v as f32 / 255.0 - 0.5) / 0.5;
        let patch_id = |i: usize| prep.patches[i * 3 * ps * ps];
        // Merge-major seq 0..4 ← patches (0,0),(0,1),(1,0),(1,1) → r = 1,2,5,6.
        assert_eq!(patch_id(0), norm(1));
        assert_eq!(patch_id(1), norm(2));
        assert_eq!(patch_id(2), norm(5));
        assert_eq!(patch_id(3), norm(6));
        // seq 4..8 ← block (by=0,bx=1): (0,2),(0,3),(1,2),(1,3) → r = 3,4,7,8.
        assert_eq!(patch_id(4), norm(3));
        assert_eq!(patch_id(7), norm(8));
        // seq 12..16 ← block (by=1,bx=1): (2,2),(2,3),(3,2),(3,3) → r = 11,12,15,16.
        assert_eq!(patch_id(12), norm(11));
        assert_eq!(patch_id(15), norm(16));

        // Within-patch layout: channel-planar. Green plane starts at ps*ps, blue at 2*ps*ps;
        // both are 0 here, and the red plane is uniform.
        let p0 = &prep.patches[0..3 * ps * ps];
        assert!(p0[..ps * ps].iter().all(|&v| v == norm(1)));
        assert!(p0[ps * ps..].iter().all(|&v| v == norm(0)));
    }

    /// Position embeddings must be reordered with the same permutation as the patches: make
    /// embd=1-equivalent identifiable positions and check merge-major seq 1 (patch (0,1)) picks
    /// the resized row-major slot for (px=1, py=0).
    #[test]
    fn pos_embed_reordered_with_patches() {
        let cfg = test_cfg();
        // base=2 grid, embd=8: give position p a signature of all-`p` so resize output slot
        // (py*nx+px) carries a recognizable (possibly interpolated) scalar per element.
        let mut pos_table = vec![0.0f32; 4 * 8];
        for p in 0..4 {
            for d in 0..8 {
                pos_table[p * 8 + d] = p as f32;
            }
        }
        let img = RgbImage::from_pixel(64, 64, Rgb([128, 128, 128]));
        let prep = prepare_image_bytes(&png_of(&img), &cfg, &pos_table).expect("prepare");
        // Row-major resized grid is 4x4 over base 2 (align corners): slot (py,px) value =
        // (py*1 + px/3-ish)... rather than recompute, verify CONSISTENCY: seq i's pos row must
        // equal the row-major row at merge_major_to_patch(i).
        let row_major = bilinear_resize_pos_table(&pos_table, 8, 2, 4, 4).expect("resize");
        for i in 0..16 {
            let (py, px) = merge_major_to_patch(i, 4, 2);
            let src = (py * 4 + px) * 8;
            assert_eq!(
                &prep.pos_embed[i * 8..(i + 1) * 8],
                &row_major[src..src + 8]
            );
        }
    }

    #[test]
    fn rejects_wrong_pos_table_len() {
        let cfg = test_cfg();
        let img = RgbImage::from_pixel(32, 32, Rgb([0, 0, 0]));
        let err = prepare_image_bytes(&png_of(&img), &cfg, &[0.0f32; 7]).map(|_| ());
        assert!(err.is_err());
    }
}
