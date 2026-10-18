//! Direct numeric parity for the IQ3_S token-embedding gather (`Op::EmbedGather`).
//!
//! The `embed_gather_iq3s` build is the only consumer of `native_decode.glsl`'s `FMT_IQ3S`
//! `dqblk`, and it is also the only grid-codebook gather: the kernel needs the 2 KiB `IQ3S_GRID`
//! staged into shared memory by `GRID_INIT` / `-DUSE_GRID` before any decode. A missing grid
//! leaves every weight decoding out of uninitialised shared memory, which no
//! Vulkan-vs-Vulkan parity test can see (both legs would share the same zeros) and which the
//! serving loop only reveals as degenerate output. So this compares the kernel against the HOST
//! dequant that the `INFR_NO_GPU_EMBED` fallback uses on a synthesized table.
//!
//! Run: `cargo test -p infr-vulkan --test iq3s_embed_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::VulkanBackend;

const IQ3S_BLOCK_ELEMS: usize = 256;
const IQ3S_BLOCK_BYTES: usize = 110;

/// `n_rows` x `ne` elements of well-formed IQ3_S blocks. Every byte after the f16 scale is free
/// (the grid indices are 9-bit into a 512-entry table, and the sign/scale bytes are arbitrary), so
/// random bytes exercise the whole codebook. The scale itself is pinned to a finite f16: the
/// `K`-quant scale lives in bytes 0..2 and an all-ones exponent would make every product NaN,
/// which a bitwise compare would then pass vacuously.
fn synth_table(n_rows: usize, ne: usize, seed: u64) -> Vec<u8> {
    let nb = ne / IQ3S_BLOCK_ELEMS;
    let blocks = n_rows * nb;
    let mut w = vec![0u8; blocks * IQ3S_BLOCK_BYTES];
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for b in 0..blocks {
        let blk = &mut w[b * IQ3S_BLOCK_BYTES..(b + 1) * IQ3S_BLOCK_BYTES];
        blk[0] = 0x00;
        blk[1] = 0x30; // f16 0.125
        for j in 2..IQ3S_BLOCK_BYTES {
            blk[j] = (next() >> 33) as u8;
        }
    }
    w
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn iq3s_embed_gather_matches_host_dequant() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (n_rows, ne) = (8usize, 2560usize);
    let dtype = DType::Iq3S;
    let (blk_elems, blk_bytes) = infr_gguf::block_layout(dtype);
    assert_eq!((blk_elems, blk_bytes), (IQ3S_BLOCK_ELEMS, IQ3S_BLOCK_BYTES));
    let row_bytes = ne / blk_elems * blk_bytes;
    let table = synth_table(n_rows, ne, 0x1234_5678_9abc_def0);
    assert_eq!(table.len(), n_rows * row_bytes);

    // Out of order, including row 0 and the highest row: a broken row-stride multiply cannot pass
    // by accident the way an all-zero/last-row-only id set would.
    let ids: [i32; 8] = [7, 0, 3, 5, 1, 6, 2, 4];
    let ids_buf = be.alloc(ids.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&ids))
        .unwrap();
    let (arena, addr) = be.alloc_arena_bda(table.len()).unwrap();
    be.upload(arena.as_ref(), &table).unwrap();

    for (leg, shader_row_bytes) in [("element-offset", 0u32), ("64-bit-row-fold", row_bytes as u32)] {
        let dst = be
            .alloc(ids.len() * ne * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        // scale = 1.0 so this is a pure decode, with no scaling of its own to confound compare.
        rec.embed_gather_at(
            dtype,
            addr,
            ids_buf.as_ref(),
            dst.as_ref(),
            ids.len(),
            ne,
            1.0,
            shader_row_bytes,
        );
        rec.finish().unwrap();

        let mut out = vec![0u8; ids.len() * ne * 4];
        be.download(dst.as_ref(), &mut out).unwrap();
        let gpu: &[f32] = bytemuck::cast_slice(&out);

        let mut mismatches = 0usize;
        let mut max_abs = 0f32;
        let mut nonzero = false;
        for (r, &id) in ids.iter().enumerate() {
            let off = id as usize * row_bytes;
            let host = infr_gguf::dequant::dequant_block(dtype, &table[off..off + row_bytes])
                .expect("host dequant of one IQ3_S embedding row");
            for i in 0..ne {
                let (g, h) = (gpu[r * ne + i], host[i]);
                let abs = (g - h).abs();
                max_abs = max_abs.max(abs);
                nonzero |= g != 0.0;
                if g.to_bits() != h.to_bits() {
                    if mismatches < 10 {
                        println!(
                            "  [{leg}] MISMATCH id {id} elem {i}: gpu {g:.9} host {h:.9}"
                        );
                    }
                    mismatches += 1;
                }
            }
        }
        println!(
            "iq3s embed_gather [{leg}]: {} rows x {ne} elements, {mismatches} not bit-identical, max_abs {max_abs:.3e}",
            ids.len()
        );
        assert!(nonzero, "[{leg}] every gathered embedding was zero — nothing ran");
        assert_eq!(
            mismatches, 0,
            "[{leg}] IQ3_S embed_gather disagrees with dequant_block (max_abs {max_abs:.3e})"
        );
    }
}
