//! Pure-CPU validation of the CHUNKED gated delta rule against the sequential recurrence —
//! the math the `deltanet_chunked` Vulkan kernel implements. No GPU required.
//!
//! Sequential (per value head, state S[kd,vd]; k̂/q̂ L2-normalized, q̂ also ×1/√kd):
//!   S ← γ_t S;  kv = k̂ᵀS;  Δ_t = β_t(v_t − kv);  S ← S + k̂⊗Δ_t;  o_t = q̂ᵀS
//! Chunked (chunk C, log-decay g_t = ln γ_t, inclusive prefix G_j = Σ_{l≤j} g_l):
//!   A[j][l] = β_j e^{G_j−G_l} (k̂_j·k̂_l)          (l<j, strictly lower)
//!   R_j     = β_j v_j − β_j e^{G_j} (k̂_jᵀ S₀)
//!   Δ_j     = R_j − Σ_{l<j} A[j][l] Δ_l            (unit-lower-triangular solve)
//!   o_j     = e^{G_j} (q̂_jᵀ S₀) + Σ_{l≤j} e^{G_j−G_l}(q̂_j·k̂_l) Δ_l
//!   S_C     = e^{G_{C−1}} S₀ + Σ_j e^{G_{C−1}−G_j} k̂_j ⊗ Δ_j

fn l2(s: &[f32], eps: f32) -> f32 {
    (s.iter().map(|x| x * x).sum::<f32>() + eps).sqrt()
}

fn softplus(z: f32) -> f32 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

#[allow(clippy::too_many_arguments)]
fn seq_delta(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: &[f32],
    a: &[f32],
    acoef: &[f32],
    dtb: &[f32],
    state: &mut [f32],
    rows: usize,
    nv: usize,
    nk: usize,
    kd: usize,
    vd: usize,
    eps: f32,
) -> Vec<f32> {
    let qscale = 1.0 / (kd as f32).sqrt();
    let mut out = vec![0f32; rows * nv * vd];
    for t in 0..rows {
        let (qb, vb, bb) = (t * nk * kd, t * nv * vd, t * nv);
        for h in 0..nv {
            let kh_idx = h % nk;
            let mut qh = q[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
            let mut kh = k[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
            let vh = &v[vb + h * vd..vb + h * vd + vd];
            let (qn, kn) = (l2(&qh, eps), l2(&kh, eps));
            for x in qh.iter_mut() {
                *x = *x / qn * qscale;
            }
            for x in kh.iter_mut() {
                *x /= kn;
            }
            let beta = 1.0 / (1.0 + (-b[bb + h]).exp());
            let decay = (acoef[h] * softplus(a[bb + h] + dtb[h])).exp();
            let sh = &mut state[h * kd * vd..(h + 1) * kd * vd];
            for x in sh.iter_mut() {
                *x *= decay;
            }
            let mut kv = vec![0f32; vd];
            for kk in 0..kd {
                for d in 0..vd {
                    kv[d] += kh[kk] * sh[kk * vd + d];
                }
            }
            let delta: Vec<f32> = (0..vd).map(|d| (vh[d] - kv[d]) * beta).collect();
            for kk in 0..kd {
                for d in 0..vd {
                    sh[kk * vd + d] += kh[kk] * delta[d];
                }
            }
            let oh = &mut out[vb + h * vd..vb + h * vd + vd];
            for kk in 0..kd {
                for d in 0..vd {
                    oh[d] += qh[kk] * sh[kk * vd + d];
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn chunk_delta(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: &[f32],
    a: &[f32],
    acoef: &[f32],
    dtb: &[f32],
    state: &mut [f32],
    rows: usize,
    nv: usize,
    nk: usize,
    kd: usize,
    vd: usize,
    eps: f32,
    chunk: usize,
) -> Vec<f32> {
    let qscale = 1.0 / (kd as f32).sqrt();
    let mut out = vec![0f32; rows * nv * vd];
    for h in 0..nv {
        let kh_idx = h % nk;
        let s0 = &mut state[h * kd * vd..(h + 1) * kd * vd]; // [kd, vd], carried across chunks
        let mut base = 0usize;
        while base < rows {
            let c = chunk.min(rows - base);
            // per-token normalized k̂/q̂, gates, inclusive prefix log-decay G
            let mut kn = vec![0f32; c * kd];
            let mut qn = vec![0f32; c * kd];
            let mut beta = vec![0f32; c];
            let mut gg = vec![0f32; c];
            let mut run = 0f32;
            for j in 0..c {
                let t = base + j;
                let qb = t * nk * kd + kh_idx * kd;
                let qh = &q[qb..qb + kd];
                let kh = &k[qb..qb + kd];
                let (qnm, knm) = (l2(qh, eps), l2(kh, eps));
                for kk in 0..kd {
                    qn[j * kd + kk] = qh[kk] / qnm * qscale;
                    kn[j * kd + kk] = kh[kk] / knm;
                }
                beta[j] = 1.0 / (1.0 + (-b[t * nv + h]).exp());
                run += acoef[h] * softplus(a[t * nv + h] + dtb[h]);
                gg[j] = run;
            }
            // A (strictly lower) and R; Δ via forward substitution
            let mut delta = vec![0f32; c * vd]; // starts as R, becomes Δ
            for j in 0..c {
                let t = base + j;
                let vh = &v[t * nv * vd + h * vd..t * nv * vd + h * vd + vd];
                let eg = gg[j].exp();
                for d in 0..vd {
                    let mut ks0 = 0f32;
                    for kk in 0..kd {
                        ks0 += kn[j * kd + kk] * s0[kk * vd + d];
                    }
                    delta[j * vd + d] = beta[j] * (vh[d] - eg * ks0);
                }
            }
            let mut aa = vec![0f32; c * c];
            for j in 1..c {
                for l in 0..j {
                    let mut dot = 0f32;
                    for kk in 0..kd {
                        dot += kn[j * kd + kk] * kn[l * kd + kk];
                    }
                    aa[j * c + l] = beta[j] * (gg[j] - gg[l]).exp() * dot;
                }
            }
            for j in 1..c {
                for l in 0..j {
                    let w = aa[j * c + l];
                    for d in 0..vd {
                        delta[j * vd + d] -= w * delta[l * vd + d];
                    }
                }
            }
            // O = e^G (Q̂ S₀) + tril(e^{Gi−Gj} Q̂K̂ᵀ) Δ  (inclusive diagonal)
            for i in 0..c {
                let t = base + i;
                let oh = &mut out[t * nv * vd + h * vd..t * nv * vd + h * vd + vd];
                let eg = gg[i].exp();
                for d in 0..vd {
                    let mut qs0 = 0f32;
                    for kk in 0..kd {
                        qs0 += qn[i * kd + kk] * s0[kk * vd + d];
                    }
                    oh[d] = eg * qs0;
                }
                for j in 0..=i {
                    let mut dot = 0f32;
                    for kk in 0..kd {
                        dot += qn[i * kd + kk] * kn[j * kd + kk];
                    }
                    let w = (gg[i] - gg[j]).exp() * dot;
                    for d in 0..vd {
                        oh[d] += w * delta[j * vd + d];
                    }
                }
            }
            // S ← e^{G_{c−1}} S₀ + Σ_j e^{G_{c−1}−G_j} k̂_j ⊗ Δ_j
            let gl = gg[c - 1];
            for kk in 0..kd {
                for d in 0..vd {
                    let mut acc = gl.exp() * s0[kk * vd + d];
                    for j in 0..c {
                        acc += (gl - gg[j]).exp() * kn[j * kd + kk] * delta[j * vd + d];
                    }
                    s0[kk * vd + d] = acc;
                }
            }
            base += c;
        }
    }
    out
}

fn gen(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 13 + salt * 7) % 29) as f32 - 14.0) * 0.05)
        .collect()
}

fn run_case(rows: usize, nv: usize, nk: usize, kd: usize, vd: usize, chunk: usize) {
    let eps = 1e-6f32;
    let q = gen(rows * nk * kd, 1);
    let k = gen(rows * nk * kd, 2);
    let v = gen(rows * nv * vd, 3);
    let b = gen(rows * nv, 4);
    let a: Vec<f32> = gen(rows * nv, 5);
    let acoef: Vec<f32> = gen(nv, 6).iter().map(|x| -x.abs() - 0.1).collect();
    let dtb = gen(nv, 7);
    let st0 = gen(nv * kd * vd, 8);
    let mut s_seq = st0.clone();
    let mut s_chn = st0;
    let want = seq_delta(
        &q, &k, &v, &b, &a, &acoef, &dtb, &mut s_seq, rows, nv, nk, kd, vd, eps,
    );
    let got = chunk_delta(
        &q, &k, &v, &b, &a, &acoef, &dtb, &mut s_chn, rows, nv, nk, kd, vd, eps, chunk,
    );
    let err = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max)
    };
    let (eo, es) = (err(&got, &want), err(&s_chn, &s_seq));
    println!("chunked-delta rows={rows} nv={nv} nk={nk} kd={kd} vd={vd} C={chunk}: out_err={eo:e} state_err={es:e}");
    assert!(eo < 1e-4, "out mismatch {eo}");
    assert!(es < 1e-4, "state mismatch {es}");
}

#[test]
fn chunked_delta_rule_matches_sequential() {
    run_case(7, 2, 1, 8, 8, 4); // tiny, partial last chunk
    run_case(97, 4, 2, 32, 16, 32); // GQA, C ∤ rows
    run_case(130, 16, 16, 128, 128, 32); // qwen35 dims
    run_case(64, 4, 2, 64, 64, 64); // single full chunk
}
