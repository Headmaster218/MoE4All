//! GPU compute kernels for the transformer ops the Llama forward needs (RMSNorm, RoPE, …),
//! plus a generic cached-kernel runner. Each op is validated against the host reference in
//! `infr-llama` (the working oracle).
//!
//! These eager runners (one submit each) are for validation; the single-command-buffer
//! resident forward (real speedup) reuses the same kernels via record APIs (added next).

use ash::vk;

use infr_core::{backend::BufferUsage, error::Result, Backend};

use super::{as_vk_buf, be, VulkanBackend};

/// Cached, reusable compute objects for one kernel. All fields are Vulkan handles (Copy).
#[derive(Clone, Copy)]
pub(crate) struct ComputeKernel {
    /// The cache key passed to [`VulkanBackend::kernel`] — the INFR_PROF_OPS auto-label: every
    /// recorder dispatch stamps its timestamp with this name (no manual stamp calls).
    pub name: &'static str,
    pub shader: vk::ShaderModule,
    pub ds_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub desc_pool: vk::DescriptorPool,
    pub n_buf: usize,
    pub push_size: u32,
    /// Hash of the SPIR-V this kernel was built from. The cache is keyed on `name` alone, so this
    /// lets [`VulkanBackend::kernel`]/[`VulkanBackend::kernel_sg`] debug-assert on a cache hit that
    /// the request's shader/layout matches the built one — a name reused with a different
    /// `spv`/`n_buf`/`push_size` would otherwise silently return a mismatched pipeline.
    pub spv_hash: u64,
}

/// Cheap non-cryptographic hash of a kernel's SPIR-V words — only used to catch a `name` reused
/// with a different shader on a cache hit (see [`ComputeKernel::spv_hash`]). Computed once per
/// build; the request side is only hashed inside the `debug_assert!`, so release builds skip it.
fn hash_spv(spv: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    spv.hash(&mut h);
    h.finish()
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn make_compute_kernel(
    device: &ash::Device,
    pcache: vk::PipelineCache,
    name: &'static str,
    spv: &[u32],
    n_buf: usize,
    push_size: u32,
    required_sg: Option<u32>,
    push_descriptor: bool,
) -> ComputeKernel {
    let shader = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spv), None)
    }
    .unwrap_or_else(|e| {
        panic!(
            "create_shader_module failed for kernel {name:?}: {e} — recoverable driver/alloc \
             failure (e.g. OUT_OF_DEVICE_MEMORY) during kernel compile; kernel build does not yet \
             thread Result (see docs/audit.md ops.rs finding 2)"
        )
    });

    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_buf)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    // PUSH_DESCRIPTOR_KHR: this set is bound via `cmd_push_descriptor_set` (recorder.rs /
    // ops.rs `run_kernel`), never `vkAllocateDescriptorSets` — the two are mutually exclusive
    // per the Vulkan spec, so the flag must match how every call site below binds it.
    let mut ds_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    if push_descriptor {
        ds_ci = ds_ci.flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
    }
    let ds_layout =
        unsafe { device.create_descriptor_set_layout(&ds_ci, None) }.unwrap_or_else(|e| {
            panic!(
                "create_descriptor_set_layout failed for kernel {name:?}: {e} — recoverable \
                 driver/alloc failure (e.g. OUT_OF_DEVICE_MEMORY) during kernel compile; kernel \
                 build does not yet thread Result (see docs/audit.md ops.rs finding 2)"
            )
        });

    let mut plinfo =
        vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&ds_layout));
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_size);
    if push_size > 0 {
        plinfo = plinfo.push_constant_ranges(std::slice::from_ref(&push_range));
    }
    let pipeline_layout =
        unsafe { device.create_pipeline_layout(&plinfo, None) }.unwrap_or_else(|e| {
            panic!(
                "create_pipeline_layout failed for kernel {name:?}: {e} — recoverable driver/alloc \
                 failure (e.g. OUT_OF_DEVICE_MEMORY) during kernel compile; kernel build does not \
                 yet thread Result (see docs/audit.md ops.rs finding 2)"
            )
        });

    let entry = c"main";
    let mut req_sz =
        vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(0);
    let mut stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(entry);
    if let Some(sz) = required_sg {
        req_sz = req_sz.required_subgroup_size(sz);
        stage = stage
            .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
            .push_next(&mut req_sz);
    }
    let pipeline = unsafe {
        device.create_compute_pipelines(
            pcache, // disk-persisted device cache (see pcache.rs); null = caching off
            &[vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout)],
            None,
        )
    }
    .unwrap_or_else(|(_, e)| {
        panic!(
            "create_compute_pipelines failed for kernel {name:?}: {e} — a device whose \
             coopmat/subgroup config doesn't match this kernel should have been filtered out by \
             capability detection before reaching here"
        )
    })[0];
    // A driver can return VK_SUCCESS with a VK_NULL_HANDLE pipeline (observed as the root cause of
    // a segfault on Intel/Mesa ANV: a coopmat pipeline whose tile size the device doesn't support).
    // Binding/dispatching that handle later is the actual crash — fail loudly HERE instead, with
    // the offending kernel named, rather than deref garbage downstream.
    assert!(
        pipeline != vk::Pipeline::null(),
        "create_compute_pipelines returned VK_SUCCESS with a null pipeline handle for kernel \
         {name:?} — likely an unsupported coopmat/subgroup config that slipped past capability \
         detection"
    );

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: n_buf as u32,
    }];
    let desc_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )
    }
    .unwrap_or_else(|e| {
        panic!(
            "create_descriptor_pool failed for kernel {name:?}: {e} — recoverable driver/alloc \
             failure (e.g. OUT_OF_DEVICE_MEMORY) during kernel compile; kernel build does not yet \
             thread Result (see docs/audit.md ops.rs finding 2)"
        )
    });

    ComputeKernel {
        name,
        shader,
        ds_layout,
        pipeline_layout,
        pipeline,
        desc_pool,
        n_buf,
        push_size,
        spv_hash: hash_spv(spv),
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn destroy_compute_kernel(device: &ash::Device, k: &ComputeKernel) {
    unsafe {
        device.destroy_descriptor_pool(k.desc_pool, None);
        device.destroy_pipeline(k.pipeline, None);
        device.destroy_pipeline_layout(k.pipeline_layout, None);
        device.destroy_descriptor_set_layout(k.ds_layout, None);
        device.destroy_shader_module(k.shader, None);
    }
}

/// Set-0 descriptor binding for the one-shot eager runners (`run_kernel` and
/// gemm.rs's eager/bench GEMMs). A [`ComputeKernel`]'s set layout carries PUSH_DESCRIPTOR_KHR
/// whenever the device has `VK_KHR_push_descriptor`, and such a layout must never be passed to
/// `vkAllocateDescriptorSets` (VUID-VkDescriptorSetAllocateInfo-pSetLayouts-00308) — so this
/// pushes the set inside the command buffer in that case and only falls back to a pooled classic
/// set (allocated up front from the kernel's pool) when the extension, and thus the flag, is
/// absent. Build with [`VulkanBackend::eager_bind`]; record with [`EagerBind::bind`].
pub(crate) struct EagerBind {
    /// `Some` iff the device lacks push descriptors (the classic-bind fallback).
    pooled_set: Option<vk::DescriptorSet>,
    /// One STORAGE_BUFFER info per binding 0..n_buf; owned here so the write structs built in
    /// [`Self::writes`] stay valid for the synchronous `one_shot` submit.
    infos: Vec<vk::DescriptorBufferInfo>,
}

impl EagerBind {
    fn writes(&self) -> Vec<vk::WriteDescriptorSet<'_>> {
        self.infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                let w = vk::WriteDescriptorSet::default()
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info));
                match self.pooled_set {
                    Some(set) => w.dst_set(set),
                    None => w, // push descriptors: dst_set is ignored by the spec
                }
            })
            .collect()
    }

    /// Record the set-0 binding into the one-shot command buffer (push or classic, per
    /// construction).
    ///
    /// # Safety
    /// `cmd` must be in the recording state and `layout` must be `k.pipeline_layout` of the
    /// kernel this binding was built for.
    pub(crate) unsafe fn bind(
        &self,
        shared: &super::VulkanShared,
        cmd: vk::CommandBuffer,
        layout: vk::PipelineLayout,
    ) {
        match (&shared.push_descriptor, self.pooled_set) {
            (Some(pd), _) => pd.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &self.writes(),
            ),
            (None, Some(set)) => shared.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            ),
            (None, None) => unreachable!("pooled_set is Some whenever push_descriptor is None"),
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VulkanBackend {
    /// Fetch-or-build a named kernel from precompiled SPIR-V (build-compiled GLSL → `.spv`).
    pub(crate) fn kernel(
        &self,
        name: &'static str,
        spv: &[u32],
        n_buf: usize,
        push_size: u32,
    ) -> ComputeKernel {
        self.kernel_inner(name, spv, n_buf, push_size, None)
    }

    /// Like `kernel`, but pins the pipeline's subgroup size (coopmat needs wave32 on RDNA3).
    /// A size of `0` means "no subgroup requirement" (treated like [`Self::kernel`]) — a caller
    /// that passes 0 by mistake would otherwise build an invalid `RequiredSubgroupSizeCreateInfo`.
    pub(crate) fn kernel_sg(
        &self,
        name: &'static str,
        spv: &[u32],
        n_buf: usize,
        push_size: u32,
        sg_size: u32,
    ) -> ComputeKernel {
        self.kernel_inner(
            name,
            spv,
            n_buf,
            push_size,
            (sg_size > 0).then_some(sg_size),
        )
    }

    /// Shared fetch-or-build for [`Self::kernel`]/[`Self::kernel_sg`]. The build runs UNDER the
    /// cache lock (via the `entry` API) so exactly one thread compiles a given `name` on
    /// concurrent first use — the old lock→get→unlock→compile→lock→insert had two racing threads
    /// both build the full shader+pipeline+pool and the second `insert` overwrite (and leak for
    /// process life) the first's Vulkan objects.
    fn kernel_inner(
        &self,
        name: &'static str,
        spv: &[u32],
        n_buf: usize,
        push_size: u32,
        required_sg: Option<u32>,
    ) -> ComputeKernel {
        let mut map = self.shared.kernels.lock().unwrap();
        let mut built = false;
        let k = *map.entry(name).or_insert_with(|| {
            built = true;
            make_compute_kernel(
                &self.shared.device,
                self.shared.pipeline_cache,
                name,
                spv,
                n_buf,
                push_size,
                required_sg,
                self.shared.push_descriptor.is_some(),
            )
        });
        drop(map);
        if built {
            self.shared.persist_pipeline_cache(); // new pipeline -> debounced disk save
        } else {
            // Cache HIT: the cache is keyed on `name` alone, so a name reused with a different
            // shader/layout would silently hand back a mismatched pipeline (VUID/UB). Fail loudly
            // in debug instead. The spv hash is only computed on this hit path in debug builds.
            debug_assert_eq!(
                k.n_buf, n_buf,
                "kernel {name:?} cached with n_buf={} but re-requested with n_buf={n_buf}",
                k.n_buf
            );
            debug_assert_eq!(
                k.push_size, push_size,
                "kernel {name:?} cached with push_size={} but re-requested with push_size={push_size}",
                k.push_size
            );
            debug_assert_eq!(
                k.spv_hash,
                hash_spv(spv),
                "kernel {name:?} cached from a different SPIR-V than this request — a name reused \
                 across two distinct shaders returns a mismatched pipeline"
            );
        }
        k
    }

    /// Build the [`EagerBind`] for one one-shot dispatch of `k`: with push descriptors the
    /// writes are recorded later by [`EagerBind::bind`]; without, a classic set is allocated
    /// from `k.desc_pool` (reset first — the pool holds one set) and written here.
    pub(crate) fn eager_bind(&self, k: &ComputeKernel, bufs: &[vk::Buffer]) -> Result<EagerBind> {
        assert_eq!(bufs.len(), k.n_buf);
        let device = &self.shared.device;
        let pooled_set = if self.shared.push_descriptor.is_none() {
            unsafe {
                device
                    .reset_descriptor_pool(k.desc_pool, vk::DescriptorPoolResetFlags::empty())
                    .map_err(|e| be(format!("reset pool: {e}")))?;
            }
            Some(unsafe {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(k.desc_pool)
                            .set_layouts(std::slice::from_ref(&k.ds_layout)),
                    )
                    .map_err(|e| be(format!("alloc set: {e}")))?[0]
            })
        } else {
            None
        };
        let bind = EagerBind {
            pooled_set,
            infos: bufs
                .iter()
                .map(|&buffer| vk::DescriptorBufferInfo {
                    buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                })
                .collect(),
        };
        if bind.pooled_set.is_some() {
            unsafe { device.update_descriptor_sets(&bind.writes(), &[]) };
        }
        Ok(bind)
    }

    /// Eagerly run a kernel: bind `inputs` (read) then one output buffer (read_write), push
    /// `push` bytes, dispatch `groups` workgroups in x. Host-visible buffers → memcpy I/O.
    fn run_kernel(
        &self,
        k: ComputeKernel,
        inputs: &[&[f32]],
        out_len: usize,
        push: &[u8],
        groups: u32,
    ) -> Result<Vec<f32>> {
        assert_eq!(inputs.len() + 1, k.n_buf);
        assert_eq!(push.len() as u32, k.push_size);

        // input buffers (host-visible) + output buffer (readback)
        let mut bufs = Vec::with_capacity(k.n_buf);
        for inp in inputs {
            let bytes: &[u8] = bytemuck::cast_slice(inp);
            let b = self.alloc(bytes.len().max(4), BufferUsage::Staging)?;
            self.upload(b.as_ref(), bytes)?;
            bufs.push(b);
        }
        let out = self.alloc((out_len * 4).max(4), BufferUsage::Readback)?;

        let vk_bufs: Vec<vk::Buffer> = bufs
            .iter()
            .map(|b| b.as_ref())
            .chain(std::iter::once(out.as_ref()))
            .map(|b| as_vk_buf(b).map(|v| v.buffer))
            .collect::<Result<Vec<_>>>()?;
        let binding = self.eager_bind(&k, &vk_bufs)?;

        let shared = &self.shared;
        self.one_shot(|cmd| unsafe {
            shared
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipeline);
            binding.bind(shared, cmd, k.pipeline_layout);
            if k.push_size > 0 {
                shared.device.cmd_push_constants(
                    cmd,
                    k.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            shared.device.cmd_dispatch(cmd, groups, 1, 1);
        })?;

        let mut out_bytes = vec![0u8; out_len * 4];
        self.download(out.as_ref(), &mut out_bytes)?;
        Ok(bytemuck::cast_slice(&out_bytes).to_vec())
    }

    /// RMSNorm over rows: `y[r,i] = x[r,i] / sqrt(mean(x[r]^2)+eps) * w[i]`.
    pub fn rmsnorm(
        &self,
        x: &[f32],
        w: &[f32],
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let k = self.kernel_sg("rmsnorm", crate::gemm::rmsnorm_spv(), 3, 12, 32);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&eps.to_ne_bytes());
        self.run_kernel(k, &[x, w], rows * dim, &push, rows as u32) // one workgroup per row
    }

    /// RoPE (ggml NORM, interleaved pairs) over `x` laid out `[t, n_heads, hd]`.
    pub fn rope(
        &self,
        x: &[f32],
        t: usize,
        n_heads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
    ) -> Result<Vec<f32>> {
        // push_size 28 must match rope.comp's full PC block (…, pos_offset, out_base) AND the
        // recorder's `kernel("rope", …, 28)` — the kernel cache is shared by name, so a smaller
        // range here would create a pipeline layout that doesn't contain the shader's block
        // (VUID-VkComputePipelineCreateInfo-layout-10069).
        let k = self.kernel("rope", crate::gemm::rope_spv(), 2, 28);
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_heads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&0u32.to_ne_bytes()); // pos_offset
        push[24..28].copy_from_slice(&0u32.to_ne_bytes()); // out_base (eager: write at row 0)
        self.run_kernel(k, &[x], t * n_heads * hd, &push, (t * n_heads) as u32)
    }

    /// SwiGLU activation: `y[i] = silu(gate[i]) * up[i]`.
    pub fn silu_mul(&self, gate: &[f32], up: &[f32], n: usize) -> Result<Vec<f32>> {
        let k = self.kernel("silu_mul", crate::gemm::silu_mul_spv(), 3, 28);
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        self.run_kernel(k, &[gate, up], n, &push, (n as u32).div_ceil(64))
    }

    /// Elementwise add: `y[i] = a[i] + b[i]`.
    pub fn add(&self, a: &[f32], b: &[f32], n: usize) -> Result<Vec<f32>> {
        let k = self.kernel("add", crate::gemm::add_spv(), 3, 4);
        self.run_kernel(
            k,
            &[a, b],
            n,
            &(n as u32).to_ne_bytes(),
            (n as u32).div_ceil(64),
        )
    }

    /// Causal GQA attention (online softmax). q `[t, nh, hd]`, k/v `[t, nkv, hd]` → `[t, nh, hd]`.
    /// Requires `hd <= 128`.
    pub fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        t: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
    ) -> Result<Vec<f32>> {
        assert!(hd <= 128, "attention kernel supports hd<=128");
        let kern = self.kernel("attention", crate::gemm::attention_spv(), 4, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        self.run_kernel(kern, &[q, k, v], t * nh * hd, &push, (t * nh) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn rmsnorm_matches_host() {
        let be = VulkanBackend::new().unwrap();
        let (rows, dim, eps) = (3usize, 8usize, 1e-5f32);
        let x: Vec<f32> = (0..rows * dim).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let w: Vec<f32> = (0..dim).map(|i| 1.0 + i as f32 * 0.05).collect();
        let got = be.rmsnorm(&x, &w, rows, dim, eps).unwrap();
        for r in 0..rows {
            let row = &x[r * dim..(r + 1) * dim];
            let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let scale = 1.0 / (ms + eps).sqrt();
            for i in 0..dim {
                let want = row[i] * scale * w[i];
                assert!(
                    (got[r * dim + i] - want).abs() < 1e-4,
                    "{} vs {}",
                    got[r * dim + i],
                    want
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn rope_matches_host() {
        let be = VulkanBackend::new().unwrap();
        let (t, nh, hd, rd, theta) = (4usize, 2usize, 8usize, 8usize, 10000.0f32);
        let x: Vec<f32> = (0..t * nh * hd).map(|i| (i as f32) * 0.03).collect();
        let got = be.rope(&x, t, nh, hd, rd, theta).unwrap();
        // host reference (ggml NORM interleaved)
        let mut want = x.clone();
        for pos in 0..t {
            for h in 0..nh {
                let base = (pos * nh + h) * hd;
                for i in 0..rd / 2 {
                    let freq = (theta as f64).powf(-2.0 * i as f64 / rd as f64) as f32;
                    let ang = pos as f32 * freq;
                    let (s, co) = ang.sin_cos();
                    let a = x[base + 2 * i];
                    let b = x[base + 2 * i + 1];
                    want[base + 2 * i] = a * co - b * s;
                    want[base + 2 * i + 1] = a * s + b * co;
                }
            }
        }
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "{g} vs {w}");
        }
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn silu_mul_matches_host() {
        let be = VulkanBackend::new().unwrap();
        let n = 100usize;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 2.0).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
        let got = be.silu_mul(&gate, &up, n).unwrap();
        for i in 0..n {
            let g = gate[i];
            let want = (g / (1.0 + (-g).exp())) * up[i];
            assert!((got[i] - want).abs() < 1e-4, "{} vs {}", got[i], want);
        }
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn add_matches_host() {
        let be = VulkanBackend::new().unwrap();
        let n = 100usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let got = be.add(&a, &b, n).unwrap();
        for i in 0..n {
            assert!((got[i] - (a[i] + b[i])).abs() < 1e-4);
        }
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_matches_host() {
        let be = VulkanBackend::new().unwrap();
        let (t, nh, nkv, hd) = (5usize, 4usize, 2usize, 8usize);
        let q: Vec<f32> = (0..t * nh * hd).map(|i| (i as f32 * 0.013).sin()).collect();
        let k: Vec<f32> = (0..t * nkv * hd)
            .map(|i| (i as f32 * 0.017).cos())
            .collect();
        let v: Vec<f32> = (0..t * nkv * hd).map(|i| i as f32 * 0.011).collect();
        let got = be.attention(&q, &k, &v, t, nh, nkv, hd).unwrap();

        // host reference: causal GQA with standard softmax
        let scale = 1.0 / (hd as f32).sqrt();
        let group = nh / nkv;
        let mut want = vec![0f32; t * nh * hd];
        for ti in 0..t {
            for h in 0..nh {
                let kvh = h / group;
                let qv = &q[(ti * nh + h) * hd..(ti * nh + h) * hd + hd];
                let mut scores = vec![0f32; ti + 1];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=ti {
                    let kv = &k[(j * nkv + kvh) * hd..(j * nkv + kvh) * hd + hd];
                    let mut d = 0f32;
                    for x in 0..hd {
                        d += qv[x] * kv[x];
                    }
                    d *= scale;
                    scores[j] = d;
                    mx = mx.max(d);
                }
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let ob = (ti * nh + h) * hd;
                for j in 0..=ti {
                    let p = scores[j] / sum;
                    let vv = &v[(j * nkv + kvh) * hd..(j * nkv + kvh) * hd + hd];
                    for x in 0..hd {
                        want[ob + x] += p * vv[x];
                    }
                }
            }
        }
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "{g} vs {w}");
        }
    }
}
