//! GPU stochastic sampler (`Recorder::sample_topk` / `Op::Sample`) vs the host reference: same
//! logits + same uniform draw must pick the same token. The reference below replicates
//! `infr-llama::sampling::sample_logits` with the draw factored out (this crate can't depend on
//! infr-llama); keep the two in sync.
//! Run: cargo test -p infr-vulkan --test sample_topk -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

fn host_sample(logits: &[f32], k: usize, temp: f32, top_p: f32, u: f32) -> u32 {
    let cmp = |a: &usize, b: &usize| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(k - 1, cmp);
    idx.truncate(k);
    idx.sort_unstable_by(cmp);
    let maxl = logits[idx[0]];
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i] - maxl) / temp).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let mut cum = 0.0;
    let mut cutoff = probs.len();
    for (j, &p) in probs.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            cutoff = j + 1;
            break;
        }
    }
    let total: f32 = probs[..cutoff].iter().sum();
    let r = u * total;
    let mut acc = 0.0;
    for j in 0..cutoff {
        acc += probs[j];
        if r <= acc {
            return idx[j] as u32;
        }
    }
    idx[cutoff - 1] as u32
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn sample_topk_matches_host() {
    let be = VulkanBackend::new().unwrap();
    let n = 151_936usize; // qwen3 vocab
                          // Deterministic pseudo-random logits (distinct values — exact ties would legitimately
                          // diverge in sort order between the two implementations).
    let mut state = 0x2545F4914F6CDD1Du64;
    let logits: Vec<f32> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 20.0 - 10.0
        })
        .collect();
    let lb = be.alloc(n * 4, BufferUsage::Activations).unwrap();
    be.upload(lb.as_ref(), bytemuck::cast_slice(&logits))
        .unwrap();
    let (top_k, temp, top_p) = (20usize, 0.6f32, 0.95f32);
    let cand = be
        .alloc(2 * 256 * top_k * 4, BufferUsage::Activations)
        .unwrap();
    let ub = be.alloc(4, BufferUsage::Staging).unwrap();
    let ob = be.alloc(4, BufferUsage::Readback).unwrap();
    for i in 0..32 {
        let u = (i as f32 + 0.5) / 32.0;
        be.upload(ub.as_ref(), &u.to_le_bytes()).unwrap();
        let rec = be.recorder().unwrap();
        rec.sample_topk(
            lb.as_ref(),
            cand.as_ref(),
            ub.as_ref(),
            ob.as_ref(),
            n,
            1,
            top_k,
            temp,
            top_p,
        );
        rec.finish().unwrap();
        let mut idb = [0u8; 4];
        be.download(ob.as_ref(), &mut idb).unwrap();
        let gpu = u32::from_le_bytes(idb);
        let host = host_sample(&logits, top_k, temp, top_p, u);
        assert_eq!(gpu, host, "u={u}: gpu {gpu} != host {host}");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn sample_topk_rows_match_host() {
    let be = VulkanBackend::new().unwrap();
    let (rows, n) = (2usize, 151_936usize);
    let mut state = 0x9E3779B97F4A7C15u64;
    let logits: Vec<f32> = (0..rows * n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 20.0 - 10.0
        })
        .collect();
    let uniforms = [0.29f32, 0.74f32];
    let (top_k, temp, top_p) = (20usize, 1.0f32, 0.95f32);
    let lb = be.alloc(rows * n * 4, BufferUsage::Activations).unwrap();
    let cand = be
        .alloc(rows * 2 * 256 * top_k * 4, BufferUsage::Activations)
        .unwrap();
    let ub = be.alloc(rows * 4, BufferUsage::Staging).unwrap();
    let ob = be.alloc(rows * 4, BufferUsage::Readback).unwrap();
    be.upload(lb.as_ref(), bytemuck::cast_slice(&logits))
        .unwrap();
    be.upload(ub.as_ref(), bytemuck::cast_slice(&uniforms))
        .unwrap();

    let rec = be.recorder().unwrap();
    rec.sample_topk(
        lb.as_ref(),
        cand.as_ref(),
        ub.as_ref(),
        ob.as_ref(),
        n,
        rows,
        top_k,
        temp,
        top_p,
    );
    rec.finish().unwrap();

    let mut ids = [0u32; 2];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut ids))
        .unwrap();
    for row in 0..rows {
        let expected = host_sample(
            &logits[row * n..(row + 1) * n],
            top_k,
            temp,
            top_p,
            uniforms[row],
        );
        assert_eq!(ids[row], expected, "row={row}");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn non_finite_logits_emit_the_corruption_sentinel() {
    let be = VulkanBackend::new().unwrap();
    let (rows, n) = (2usize, 4096usize);
    let mut logits = vec![-5.0f32; rows * n];
    logits[17] = 9.0;
    logits[n + 23] = 10.0;
    logits[n + 701] = f32::NAN;

    let lb = be
        .alloc(logits.len() * 4, BufferUsage::Activations)
        .unwrap();
    be.upload(lb.as_ref(), bytemuck::cast_slice(&logits))
        .unwrap();

    let argmax_part = be.alloc(512 * 4, BufferUsage::Activations).unwrap();
    let argmax_out = be.alloc(rows * 4, BufferUsage::Readback).unwrap();
    let rec = be.recorder().unwrap();
    rec.argmax(
        lb.as_ref(),
        argmax_part.as_ref(),
        argmax_out.as_ref(),
        n,
        rows,
    );
    rec.finish().unwrap();
    let mut argmax_ids = [0u32; 2];
    be.download(
        argmax_out.as_ref(),
        bytemuck::cast_slice_mut(&mut argmax_ids),
    )
    .unwrap();
    assert_eq!(argmax_ids, [17, u32::MAX]);

    let top_k = 20usize;
    let cand = be
        .alloc(rows * 2 * 256 * top_k * 4, BufferUsage::Activations)
        .unwrap();
    let uniforms = [0.25f32, 0.75f32];
    let ub = be.alloc(rows * 4, BufferUsage::Staging).unwrap();
    let sample_out = be.alloc(rows * 4, BufferUsage::Readback).unwrap();
    be.upload(ub.as_ref(), bytemuck::cast_slice(&uniforms))
        .unwrap();
    let rec = be.recorder().unwrap();
    rec.sample_topk(
        lb.as_ref(),
        cand.as_ref(),
        ub.as_ref(),
        sample_out.as_ref(),
        n,
        rows,
        top_k,
        0.8,
        0.95,
    );
    rec.finish().unwrap();
    let mut sample_ids = [0u32; 2];
    be.download(
        sample_out.as_ref(),
        bytemuck::cast_slice_mut(&mut sample_ids),
    )
    .unwrap();
    assert_ne!(sample_ids[0], u32::MAX);
    assert_eq!(sample_ids[1], u32::MAX);

    let bad_row = be.alloc(n * 4, BufferUsage::Activations).unwrap();
    be.upload(bad_row.as_ref(), bytemuck::cast_slice(&logits[n..2 * n]))
        .unwrap();
    let prob_part = be.alloc(768 * 4, BufferUsage::Activations).unwrap();
    let prob_id = be.alloc(4, BufferUsage::Readback).unwrap();
    let prob = be.alloc(4, BufferUsage::Readback).unwrap();
    let rec = be.recorder().unwrap();
    rec.argmax_prob(
        bad_row.as_ref(),
        prob_part.as_ref(),
        prob_id.as_ref(),
        prob.as_ref(),
        n,
    );
    rec.finish().unwrap();
    let mut id = [0u8; 4];
    be.download(prob_id.as_ref(), &mut id).unwrap();
    assert_eq!(u32::from_le_bytes(id), u32::MAX);
}
