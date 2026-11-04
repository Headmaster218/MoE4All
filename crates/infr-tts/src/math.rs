//! Tensor layout is [time, channels]. All dense products, including convolution
//! im2col and attention, execute through the existing INFR CPU F32 kernels.
use anyhow::{ensure, Result};
use infr_core::{Backend, Buffer, BufferUsage};
#[cfg(not(feature = "borrowed-f32"))]
use infr_core::{Bindings, DType, Graph, Op, TensorDesc};
use infr_cpu::CpuBackend;
use std::sync::atomic::AtomicBool;

pub(crate) const TILE: usize = 128;

#[cfg(test)]
thread_local! {static CONV_TIMES:std::cell::Cell<(u128,u128)>=const{std::cell::Cell::new((0,0))};}
#[cfg(test)]
pub(crate) fn take_conv_times() -> (f64, f64) {
    CONV_TIMES.with(|v| {
        let (packing, linear) = v.replace((0, 0));
        (packing as f64 / 1e6, linear as f64 / 1e6)
    })
}

#[derive(Clone, Debug)]
pub(crate) struct Signal {
    pub t: usize,
    pub c: usize,
    pub v: Vec<f32>,
}
impl Signal {
    pub fn zeros(t: usize, c: usize) -> Self {
        Self {
            t,
            c,
            v: vec![0.; t * c],
        }
    }
    pub fn add(&mut self, other: &Self) -> Result<()> {
        #[cfg(feature = "cpu-ab-profile")]
        let _phase = crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::Add);
        ensure!(
            self.t == other.t && self.c == other.c,
            "residual shape mismatch"
        );
        for (a, b) in self.v.iter_mut().zip(&other.v) {
            *a += *b;
        }
        Ok(())
    }
    pub fn bias(&mut self, bias: &[f32]) -> Result<()> {
        #[cfg(feature = "cpu-ab-profile")]
        let _phase = crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::Bias);
        ensure!(bias.len() == self.c, "bias shape mismatch");
        for row in self.v.chunks_exact_mut(self.c) {
            for (a, b) in row.iter_mut().zip(bias) {
                *a += *b;
            }
        }
        Ok(())
    }
    pub fn leaky(&mut self, slope: f32) {
        #[cfg(feature = "cpu-ab-profile")]
        let _phase = crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::Leaky);
        for a in &mut self.v {
            if *a < 0. {
                *a *= slope;
            }
        }
    }
    #[cfg(feature = "fused-conv")]
    pub fn clone_leaky(&self, slope: f32) -> Self {
        #[cfg(feature = "cpu-ab-profile")]
        let _phase = crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::Leaky);
        Self {
            t: self.t,
            c: self.c,
            v: self
                .v
                .iter()
                .map(|&v| if v < 0. { v * slope } else { v })
                .collect(),
        }
    }
    pub fn relu(&mut self) {
        for a in &mut self.v {
            *a = a.max(0.);
        }
    }
}

pub(crate) fn upload(backend: &CpuBackend, values: &[f32]) -> Result<Box<dyn Buffer>> {
    let b = backend.alloc(values.len() * 4, BufferUsage::Activations)?;
    backend.upload(b.as_ref(), bytemuck::cast_slice(values))?;
    Ok(b)
}

pub(crate) fn linear(
    backend: &CpuBackend,
    input: &[f32],
    rows: usize,
    input_c: usize,
    weight: &dyn Buffer,
    output_c: usize,
) -> Result<Vec<f32>> {
    #[cfg(feature = "cpu-ab-profile")]
    let _profile = infr_cpu::f32_profile::LinearTimer::start();
    ensure!(
        rows > 0 && input_c > 0 && output_c > 0,
        "empty matrix product"
    );
    ensure!(input.len() == rows * input_c, "linear input shape mismatch");
    #[cfg(feature = "borrowed-f32")]
    {
        let mut values = vec![0f32; rows * output_c];
        backend.linear_f32_into(input, rows, input_c, weight, output_c, &mut values)?;
        Ok(values)
    }
    #[cfg(not(feature = "borrowed-f32"))]
    {
        let xbuf = upload(backend, input)?;
        let obuf = backend.alloc(rows * output_c * 4, BufferUsage::Activations)?;
        let mut graph = Graph::new();
        let x = graph.input(TensorDesc::new(vec![rows, input_c], DType::F32));
        let w = graph.weight(TensorDesc::new(vec![output_c, input_c], DType::F32));
        let y = graph.output(TensorDesc::new(vec![rows, output_c], DType::F32));
        graph.push(Op::Linear {
            x,
            weight: w,
            dst: y,
            m: rows as u32,
            in_f: input_c as u32,
            out_f: output_c as u32,
            w_off: 0,
        });
        let plan = backend.compile(&graph)?;
        let mut bindings = Bindings::new();
        bindings
            .bind(x, xbuf.as_ref())
            .bind(w, weight)
            .bind(y, obuf.as_ref());
        backend.execute(plan.as_ref(), &bindings)?;
        backend.sync()?;
        let mut values = vec![0f32; rows * output_c];
        backend.download(obuf.as_ref(), bytemuck::cast_slice_mut(&mut values))?;
        Ok(values)
    }
}

pub(crate) fn norm(x: &mut Signal, gamma: &[f32], beta: &[f32]) -> Result<()> {
    ensure!(
        gamma.len() == x.c && beta.len() == x.c,
        "layer norm shape mismatch"
    );
    for row in x.v.chunks_exact_mut(x.c) {
        let mean = row.iter().sum::<f32>() / x.c as f32;
        let var = row.iter().map(|a| (*a - mean) * (*a - mean)).sum::<f32>() / x.c as f32;
        let scale = 1. / (var + 1e-5).sqrt();
        for i in 0..x.c {
            row[i] = (row[i] - mean) * scale * gamma[i] + beta[i];
        }
    }
    Ok(())
}

/// PyTorch Conv1d cross-correlation, zero padded, output time unchanged.
/// Weight columns are packed [tap,channel] at load for contiguous im2col copies.
pub(crate) fn conv(
    backend: &CpuBackend,
    x: &Signal,
    weight: &dyn Buffer,
    output_c: usize,
    kernel: usize,
    dilation: usize,
    bias: Option<&[f32]>,
    cancel: &AtomicBool,
) -> Result<Signal> {
    #[cfg(feature = "fused-conv")]
    {
        return conv_fused(
            backend, x, weight, output_c, kernel, dilation, bias, None, None, cancel,
        );
    }
    #[cfg(not(feature = "fused-conv"))]
    conv_unfused(backend, x, weight, output_c, kernel, dilation, bias, cancel)
}

#[cfg(any(not(feature = "fused-conv"), test))]
fn conv_unfused(
    backend: &CpuBackend,
    x: &Signal,
    weight: &dyn Buffer,
    output_c: usize,
    kernel: usize,
    dilation: usize,
    bias: Option<&[f32]>,
    cancel: &AtomicBool,
) -> Result<Signal> {
    #[cfg(feature = "cpu-ab-profile")]
    let _phase = crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::Conv);
    ensure!(
        kernel > 0 && kernel % 2 == 1 && dilation > 0,
        "unsupported convolution geometry"
    );
    let mut out = Signal::zeros(x.t, output_c);
    let pad = dilation * (kernel - 1) / 2;
    // SGEMM packs/cache-blocks internally. A bounded 1024-row outer tile
    // amortizes Graph allocation and pool handoff without materializing the
    // whole waveform's im2col matrix. Largest supported window is ~11 MiB.
    let tile = 1024;
    for start in (0..x.t).step_by(tile) {
        crate::check_cancel(cancel)?;
        let n = tile.min(x.t - start);
        #[cfg(test)]
        let packing_start = std::time::Instant::now();
        #[cfg(feature = "cpu-ab-profile")]
        let phase = crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::ConvPack);
        let input = if kernel == 1 {
            x.v[start * x.c..(start + n) * x.c].to_vec()
        } else {
            let mut col = vec![0.; n * x.c * kernel];
            for row in 0..n {
                for tap in 0..kernel {
                    let at = (start + row + tap * dilation) as isize - pad as isize;
                    if at >= 0 && (at as usize) < x.t {
                        let destination = (row * kernel + tap) * x.c;
                        let source = at as usize * x.c;
                        col[destination..destination + x.c]
                            .copy_from_slice(&x.v[source..source + x.c]);
                    }
                }
            }
            col
        };
        #[cfg(feature = "cpu-ab-profile")]
        drop(phase);
        #[cfg(test)]
        let packing_ns = packing_start.elapsed().as_nanos();
        #[cfg(test)]
        let linear_start = std::time::Instant::now();
        #[cfg(feature = "cpu-ab-profile")]
        let phase = crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::ConvLinear);
        let y = linear(backend, &input, n, x.c * kernel, weight, output_c)?;
        #[cfg(feature = "cpu-ab-profile")]
        drop(phase);
        #[cfg(test)]
        CONV_TIMES.with(|v| {
            let (p, l) = v.get();
            v.set((p + packing_ns, l + linear_start.elapsed().as_nanos()));
        });
        #[cfg(feature = "cpu-ab-profile")]
        let _copy_phase = crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::ConvCopy);
        out.v[start * output_c..(start + n) * output_c].copy_from_slice(&y);
    }
    if let Some(b) = bias {
        out.bias(b)?;
    }
    Ok(out)
}

/// Same SGEMM tile shapes and accumulation order as conv_unfused. One bounded
/// im2col window is overwritten/reused; SGEMM writes the final output slice.
/// The elementwise order remains bias -> optional leaky -> optional residual.
#[cfg(feature = "fused-conv")]
pub(crate) fn conv_fused(
    backend: &CpuBackend,
    x: &Signal,
    weight: &dyn Buffer,
    output_c: usize,
    kernel: usize,
    dilation: usize,
    bias: Option<&[f32]>,
    slope: Option<f32>,
    residual: Option<&Signal>,
    cancel: &AtomicBool,
) -> Result<Signal> {
    #[cfg(feature = "cpu-ab-profile")]
    let _phase = crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::Conv);
    ensure!(
        kernel > 0 && kernel % 2 == 1 && dilation > 0,
        "unsupported convolution geometry"
    );
    ensure!(
        bias.is_none_or(|b| b.len() == output_c),
        "bias shape mismatch"
    );
    ensure!(
        residual.is_none_or(|r| r.t == x.t && r.c == output_c && r.v.len() == x.t * output_c),
        "residual shape mismatch"
    );
    let mut out = Signal::zeros(x.t, output_c);
    let pad = dilation * (kernel - 1) / 2;
    let tile = 1024;
    let width = x.c * kernel;
    let mut col = if kernel == 1 {
        Vec::new()
    } else {
        vec![0.; tile.min(x.t) * width]
    };
    for start in (0..x.t).step_by(tile) {
        crate::check_cancel(cancel)?;
        let n = tile.min(x.t - start);
        #[cfg(test)]
        let packing_start = std::time::Instant::now();
        #[cfg(feature = "cpu-ab-profile")]
        let phase = crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::ConvPack);
        let input = if kernel == 1 {
            &x.v[start * x.c..(start + n) * x.c]
        } else {
            let window = &mut col[..n * width];
            for (row, destination) in window.chunks_exact_mut(width).enumerate() {
                let center = start + row;
                // Most rows need no bounds test/zero-fill per tap. Boundary
                // windows overwrite every tap too, so no stale workspace data.
                if center >= pad && center + pad < x.t {
                    for (tap, target) in destination.chunks_exact_mut(x.c).enumerate() {
                        let source = (center - pad + tap * dilation) * x.c;
                        target.copy_from_slice(&x.v[source..source + x.c]);
                    }
                } else {
                    for (tap, target) in destination.chunks_exact_mut(x.c).enumerate() {
                        let at = (center + tap * dilation) as isize - pad as isize;
                        if at >= 0 && (at as usize) < x.t {
                            let source = at as usize * x.c;
                            target.copy_from_slice(&x.v[source..source + x.c]);
                        } else {
                            target.fill(0.);
                        }
                    }
                }
            }
            &*window
        };
        #[cfg(feature = "cpu-ab-profile")]
        drop(phase);
        #[cfg(test)]
        let packing_ns = packing_start.elapsed().as_nanos();
        #[cfg(test)]
        let linear_start = std::time::Instant::now();
        {
            #[cfg(feature = "cpu-ab-profile")]
            let _phase = crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::ConvLinear);
            #[cfg(feature = "cpu-ab-profile")]
            let _linear = infr_cpu::f32_profile::LinearTimer::start();
            backend.linear_f32_into(
                input,
                n,
                width,
                weight,
                output_c,
                &mut out.v[start * output_c..(start + n) * output_c],
            )?;
        }
        #[cfg(test)]
        CONV_TIMES.with(|v| {
            let (p, l) = v.get();
            v.set((p + packing_ns, l + linear_start.elapsed().as_nanos()));
        });
        if bias.is_some() || slope.is_some() || residual.is_some() {
            #[cfg(feature = "cpu-ab-profile")]
            let _phase = crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::ConvEpilogue);
            for (row, output) in out.v[start * output_c..(start + n) * output_c]
                .chunks_exact_mut(output_c)
                .enumerate()
            {
                for (channel, value) in output.iter_mut().enumerate() {
                    if let Some(b) = bias {
                        *value += b[channel];
                    }
                    if let Some(slope) = slope {
                        if *value < 0. {
                            *value *= slope;
                        }
                    }
                    if let Some(r) = residual {
                        *value = r.v[(start + row) * output_c + channel] + *value;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Packed weight is [output_channels*kernel,input_channels]. Scatter implements
/// ConvTranspose1d exactly; no inserted-zero tensor or runtime dependency.
pub(crate) fn transpose_conv(
    backend: &CpuBackend,
    x: &Signal,
    weight: &dyn Buffer,
    output_c: usize,
    kernel: usize,
    stride: usize,
    bias: &[f32],
    cancel: &AtomicBool,
) -> Result<Signal> {
    #[cfg(feature = "cpu-ab-profile")]
    let _phase = crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::Transpose);
    ensure!(
        stride > 0 && kernel >= stride && (kernel - stride) % 2 == 0,
        "invalid transpose convolution geometry"
    );
    let pad = (kernel - stride) / 2;
    let out_t = (x.t - 1) * stride + kernel - 2 * pad;
    let mut out = Signal::zeros(out_t, output_c);
    for start in (0..x.t).step_by(TILE) {
        crate::check_cancel(cancel)?;
        let n = TILE.min(x.t - start);
        #[cfg(feature = "cpu-ab-profile")]
        let phase = crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::TransposeLinear);
        let y = linear(
            backend,
            &x.v[start * x.c..(start + n) * x.c],
            n,
            x.c,
            weight,
            output_c * kernel,
        )?;
        #[cfg(feature = "cpu-ab-profile")]
        drop(phase);
        #[cfg(feature = "cpu-ab-profile")]
        let _scatter_phase =
            crate::cpu_profile::Timer::start(crate::cpu_profile::Phase::TransposeScatter);
        for row in 0..n {
            for tap in 0..kernel {
                let at = ((start + row) * stride + tap) as isize - pad as isize;
                if at < 0 || at as usize >= out_t {
                    continue;
                }
                for ch in 0..output_c {
                    out.v[at as usize * output_c + ch] += y[(row * output_c + ch) * kernel + tap];
                }
            }
        }
    }
    out.bias(bias)?;
    Ok(out)
}

/// Relative-key/value attention with a global mask and local relative window.
/// Tiles query rows to bound scratch at O(T*TILE), rather than O(T^2).
pub(crate) fn attention(
    backend: &CpuBackend,
    q: &Signal,
    k: &Signal,
    v: &Signal,
    heads: usize,
    rel_k: &[f32],
    rel_v: &[f32],
    rel_heads: usize,
    window: usize,
    cancel: &AtomicBool,
) -> Result<Signal> {
    ensure!(
        q.c % heads == 0 && q.t == k.t && q.t == v.t && q.c == k.c && q.c == v.c,
        "attention shape mismatch"
    );
    let d = q.c / heads;
    ensure!(
        (rel_heads == 1 || rel_heads == heads)
            && rel_k.len() == rel_heads * (2 * window + 1) * d
            && rel_v.len() == rel_k.len(),
        "relative attention weight shape mismatch"
    );
    let scale = 1. / (d as f32).sqrt();
    let mut out = Signal::zeros(q.t, q.c);
    for h in 0..heads {
        let mut kw = vec![0.; q.t * d];
        let mut vw = vec![0.; d * q.t];
        for t in 0..q.t {
            for c in 0..d {
                kw[t * d + c] = k.v[t * k.c + h * d + c];
                vw[c * q.t + t] = v.v[t * v.c + h * d + c];
            }
        }
        let kb = upload(backend, &kw)?;
        let vb = upload(backend, &vw)?;
        let rh = if rel_heads == 1 { 0 } else { h };
        for start in (0..q.t).step_by(TILE) {
            crate::check_cancel(cancel)?;
            let n = TILE.min(q.t - start);
            let mut query = vec![0.; n * d];
            for r in 0..n {
                query[r * d..(r + 1) * d].copy_from_slice(
                    &q.v[(start + r) * q.c + h * d..(start + r) * q.c + (h + 1) * d],
                );
            }
            let mut p = linear(backend, &query, n, d, kb.as_ref(), q.t)?;
            for r in 0..n {
                let row = &mut p[r * q.t..(r + 1) * q.t];
                for a in row.iter_mut() {
                    *a *= scale;
                }
                let i = start + r;
                for j in i.saturating_sub(window)..=(i + window).min(q.t - 1) {
                    let off = (rh * (2 * window + 1) + j + window - i) * d;
                    let mut score = 0.;
                    for c in 0..d {
                        score += query[r * d + c] * rel_k[off + c];
                    }
                    row[j] += score * scale;
                }
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.;
                for a in row.iter_mut() {
                    *a = (*a - max).exp();
                    sum += *a;
                }
                ensure!(sum.is_finite() && sum > 0., "nonfinite attention scores");
                for a in row.iter_mut() {
                    *a /= sum;
                }
            }
            let mut values = linear(backend, &p, n, q.t, vb.as_ref(), d)?;
            for r in 0..n {
                let i = start + r;
                for j in i.saturating_sub(window)..=(i + window).min(q.t - 1) {
                    let off = (rh * (2 * window + 1) + j + window - i) * d;
                    for c in 0..d {
                        values[r * d + c] += p[r * q.t + j] * rel_v[off + c];
                    }
                }
                out.v[i * q.c + h * d..i * q.c + (h + 1) * d]
                    .copy_from_slice(&values[r * d..(r + 1) * d]);
            }
        }
    }
    Ok(out)
}

pub(crate) struct Gaussian {
    state: u64,
    spare: Option<f32>,
}
impl Gaussian {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e3779b97f4a7c15,
            spare: None,
        }
    }
    fn uniform(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 40) as f32 + 0.5) / 16777216.
    }
    pub fn next(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let r = (-2. * self.uniform().ln()).sqrt();
        let theta = std::f32::consts::TAU * self.uniform();
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn backend() -> CpuBackend {
        let backend = CpuBackend::new_with_thread_limit(
            std::sync::Arc::new(infr_core::config::Config::default()),
            1,
        )
        .unwrap();
        #[cfg(feature = "borrowed-f32")]
        let backend = backend.with_f32_gemm().unwrap();
        backend
    }
    #[test]
    fn convolution_and_transpose_match_hand_computed() {
        let b = backend();
        let cancel = AtomicBool::new(false);
        let x = Signal {
            t: 3,
            c: 1,
            v: vec![1., 2., 3.],
        };
        let w = upload(&b, &[1., 2., 3.]).unwrap();
        let y = conv(&b, &x, w.as_ref(), 1, 3, 1, None, &cancel).unwrap();
        assert_eq!(y.v, vec![8., 14., 8.]);
        let channels = Signal {
            t: 3,
            c: 2,
            v: vec![1., 10., 2., 20., 3., 30.],
        };
        let packed = upload(&b, &[1., 4., 2., 5., 3., 6.]).unwrap();
        let y = conv(&b, &channels, packed.as_ref(), 1, 3, 1, None, &cancel).unwrap();
        assert_eq!(y.v, vec![178., 334., 238.]);
        let w = upload(&b, &[1., 2., 3., 4.]).unwrap();
        let y = transpose_conv(&b, &x, w.as_ref(), 1, 4, 2, &[0.5], &cancel).unwrap();
        assert_eq!(y.v, vec![2.5, 5.5, 8.5, 9.5, 14.5, 9.5]);
    }
    #[test]
    fn relative_attention_is_global_and_uses_j_minus_i() {
        let b = backend();
        let cancel = AtomicBool::new(false);
        let q = Signal {
            t: 3,
            c: 1,
            v: vec![0.; 3],
        };
        let v = Signal {
            t: 3,
            c: 1,
            v: vec![1., 2., 3.],
        };
        let y = attention(&b, &q, &q, &v, 1, &[0.; 3], &[3., 6., 9.], 1, 1, &cancel).unwrap();
        for (a, e) in y.v.iter().zip([7., 8., 5.]) {
            assert!((*a - e).abs() < 1e-5, "{a} != {e}");
        }
    }
    #[test]
    fn cancelled_convolution_returns_error() {
        let b = backend();
        let w = upload(&b, &[1.]).unwrap();
        assert!(conv(
            &b,
            &Signal {
                t: 1,
                c: 1,
                v: vec![1.]
            },
            w.as_ref(),
            1,
            1,
            1,
            None,
            &AtomicBool::new(true)
        )
        .is_err());
    }
    #[test]
    #[cfg(feature = "fused-conv")]
    fn fused_windows_epilogues_match_unfused_at_padding_and_tile_tails() {
        let backend = backend();
        let cancel = AtomicBool::new(false);
        for (time, channels, output, kernel, dilation) in [
            (1, 2, 3, 11, 5),
            (7, 3, 4, 3, 1),
            (19, 4, 8, 7, 3),
            (1027, 4, 8, 11, 5),
            (1025, 4, 8, 1, 1),
        ] {
            let x = Signal {
                t: time,
                c: channels,
                v: (0..time * channels)
                    .map(|i| ((i * 17 % 41) as f32 - 20.) / 19.)
                    .collect(),
            };
            let values: Vec<_> = (0..output * channels * kernel)
                .map(|i| ((i * 13 % 37) as f32 - 18.) / 43.)
                .collect();
            let weight = upload(&backend, &values).unwrap();
            let bias: Vec<_> = (0..output).map(|i| i as f32 / 7. - 0.3).collect();
            let residual = Signal {
                t: time,
                c: output,
                v: (0..time * output)
                    .map(|i| ((i * 7 % 19) as f32 - 9.) / 31.)
                    .collect(),
            };
            for (bias, slope, skip) in [
                (None, None, None),
                (Some(bias.as_slice()), Some(0.1), None),
                (Some(bias.as_slice()), None, Some(&residual)),
            ] {
                let mut expected = conv_unfused(
                    &backend,
                    &x,
                    weight.as_ref(),
                    output,
                    kernel,
                    dilation,
                    bias,
                    &cancel,
                )
                .unwrap();
                if let Some(slope) = slope {
                    expected.leaky(slope);
                }
                if let Some(skip) = skip {
                    let mut sum = skip.clone();
                    sum.add(&expected).unwrap();
                    expected = sum;
                }
                let actual = conv_fused(
                    &backend,
                    &x,
                    weight.as_ref(),
                    output,
                    kernel,
                    dilation,
                    bias,
                    slope,
                    skip,
                    &cancel,
                )
                .unwrap();
                assert_eq!(
                    actual.v, expected.v,
                    "geometry {time}/{channels}/{output}/{kernel}/{dilation}"
                );
            }
        }
        let x = Signal {
            t: 1,
            c: 1,
            v: vec![1.],
        };
        let w = upload(&backend, &[1.]).unwrap();
        assert!(conv_fused(
            &backend,
            &x,
            w.as_ref(),
            1,
            1,
            1,
            Some(&[]),
            None,
            None,
            &cancel
        )
        .is_err());
        assert!(conv_fused(
            &backend,
            &x,
            w.as_ref(),
            1,
            1,
            1,
            None,
            None,
            Some(&Signal::zeros(2, 1)),
            &cancel
        )
        .is_err());
    }
}
