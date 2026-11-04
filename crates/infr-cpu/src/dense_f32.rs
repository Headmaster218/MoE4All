//! The same opt-in F32 dispatch for Graph and borrowed host activations.
//! No alternate kernels, thread budget or accumulation order are introduced.
use super::{dot, dot_f32_batch, Buffer, CpuBackend, CpuBuffer, CpuStore, Error, Result};

impl CpuBackend {
    /// Row-major X[rows,input] * W[output,input]^T into row-major destination.
    /// Requires explicit `with_f32_gemm()` capability. Weights must be F32 CPU
    /// mapped/owned buffers, not paged or GPU allocations. Buffers are borrowed
    /// only until this synchronous call completes; no shape/activation cache.
    pub fn linear_f32_into(
        &self,
        x: &[f32],
        rows: usize,
        input: usize,
        weight: &dyn Buffer,
        output: usize,
        destination: &mut [f32],
    ) -> Result<()> {
        if !self.dense_f32_gemm {
            return Err(Error::backend(
                "borrowed F32 linear requires an explicitly enabled dense CPU capability",
            ));
        }
        let x_len = rows
            .checked_mul(input)
            .ok_or_else(|| Error::backend("F32 input dimensions overflow"))?;
        let y_len = rows
            .checked_mul(output)
            .ok_or_else(|| Error::backend("F32 output dimensions overflow"))?;
        let w_bytes = output
            .checked_mul(input)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::backend("F32 weight dimensions overflow"))?;
        if rows == 0 || input == 0 || output == 0 || x.len() != x_len || destination.len() != y_len
        {
            return Err(Error::backend(
                "borrowed F32 linear activation shape mismatch",
            ));
        }
        let buffer = weight
            .as_any()
            .downcast_ref::<CpuBuffer>()
            .ok_or_else(|| Error::backend("borrowed F32 linear requires a CPU weight buffer"))?;
        if matches!(&buffer.store, CpuStore::Paged { .. }) {
            return Err(Error::backend(
                "borrowed F32 linear does not establish paged weight residency",
            ));
        }
        let bytes = buffer.read();
        if bytes.len() != w_bytes {
            return Err(Error::backend("borrowed F32 linear weight shape mismatch"));
        }
        let weights: &[f32] = bytemuck::try_cast_slice(&bytes)
            .map_err(|_| Error::backend("borrowed F32 linear weight alignment mismatch"))?;
        self.compute_dense_f32(x, weights, rows, input, output, destination);
        Ok(())
    }

    pub(super) fn compute_dense_f32(
        &self,
        x: &[f32],
        weights: &[f32],
        m: usize,
        input: usize,
        output: usize,
        out: &mut [f32],
    ) {
        #[cfg(feature = "f32-gemm-profile")]
        let _profile = crate::f32_profile::DispatchTimer::start();
        #[cfg(feature = "f32-gemm-profile")]
        let task_ns = std::sync::atomic::AtomicU64::new(0);
        if !self.cfg.kernels.cpu.reference && m >= 8 && output >= 8 {
            if m >= output {
                let rows_per_job = m.div_ceil(self.dense_f32_threads).div_ceil(8) * 8;
                self.pool()
                    .for_chunks_mut(out, rows_per_job * output, 1, &|block, chunk| {
                        let first = block * rows_per_job;
                        let rows = chunk.len() / output;
                        #[cfg(feature = "f32-gemm-profile")]
                        let started = std::time::Instant::now();
                        // SAFETY: validated row-major matrices, disjoint output
                        // rows; B is the read-only transpose of row-major weights.
                        unsafe {
                            matrixmultiply::sgemm(
                                rows,
                                input,
                                output,
                                1.,
                                x[first * input..].as_ptr(),
                                input as isize,
                                1,
                                weights.as_ptr(),
                                1,
                                input as isize,
                                0.,
                                chunk.as_mut_ptr(),
                                output as isize,
                                1,
                            )
                        };
                        #[cfg(feature = "f32-gemm-profile")]
                        task_ns.fetch_add(
                            started.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    });
            } else {
                let mut out_t = vec![0f32; output * m];
                let block_rows = output.div_ceil(self.dense_f32_threads).div_ceil(8) * 8;
                self.pool()
                    .for_chunks_mut(&mut out_t, block_rows * m, 1, &|block, chunk| {
                        let first = block * block_rows;
                        let count = chunk.len() / m;
                        #[cfg(feature = "f32-gemm-profile")]
                        let started = std::time::Instant::now();
                        // SAFETY: disjoint transposed output blocks; input and
                        // weight strides select only complete, validated rows.
                        unsafe {
                            matrixmultiply::sgemm(
                                count,
                                input,
                                m,
                                1.,
                                weights[first * input..].as_ptr(),
                                input as isize,
                                1,
                                x.as_ptr(),
                                1,
                                input as isize,
                                0.,
                                chunk.as_mut_ptr(),
                                m as isize,
                                1,
                            )
                        };
                        #[cfg(feature = "f32-gemm-profile")]
                        task_ns.fetch_add(
                            started.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    });
                self.pool().for_chunks_mut(out, output, 1, &|r, row| {
                    for (o, dst) in row.iter_mut().enumerate() {
                        *dst = out_t[o * m + r];
                    }
                });
            }
            #[cfg(feature = "f32-gemm-profile")]
            crate::f32_profile::add_sgemm_task_ns(
                task_ns.load(std::sync::atomic::Ordering::Relaxed),
            );
        } else if m == 1 {
            self.pool().for_chunks_mut(out, 1, 16, &|o, value| {
                value[0] = dot(&weights[o * input..(o + 1) * input], &x[..input]);
            });
        } else {
            let mut out_t = vec![0f32; output * m];
            self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                dot_f32_batch(&weights[o * input..(o + 1) * input], x, input, chunk);
            });
            self.pool().for_chunks_mut(out, output, 1, &|r, row| {
                for (o, dst) in row.iter_mut().enumerate() {
                    *dst = out_t[o * m + r];
                }
            });
        }
    }
}
