use std::sync::{Arc, Mutex};

use ash::vk;
use infr_core::timeline::{self, Context, DeviceEvent};

use crate::{be, VulkanShared};

const QUERY_COUNT: u32 = 128;

pub(crate) struct Clock {
    loader: ash::ext::calibrated_timestamps::Device,
    domain: vk::TimeDomainEXT,
    last_step: std::sync::atomic::AtomicU64,
}

impl Clock {
    pub(crate) fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
    ) -> Option<Self> {
        let loader = ash::ext::calibrated_timestamps::Instance::new(entry, instance);
        let domains =
            unsafe { loader.get_physical_device_calibrateable_time_domains(physical) }.ok()?;
        let domain = if cfg!(windows) {
            vk::TimeDomainEXT::QUERY_PERFORMANCE_COUNTER
        } else {
            vk::TimeDomainEXT::CLOCK_MONOTONIC
        };
        if !domains.contains(&domain) || !domains.contains(&vk::TimeDomainEXT::DEVICE) {
            return None;
        }
        Some(Self {
            loader: ash::ext::calibrated_timestamps::Device::new(instance, device),
            domain,
            last_step: std::sync::atomic::AtomicU64::new(u64::MAX),
        })
    }

    pub(crate) fn sample(&self, period: f32, step: u64) {
        use std::sync::atomic::Ordering;
        if self.last_step.swap(step, Ordering::Relaxed) == step {
            return;
        }
        let _span = timeline::span("trace_clock_calibration");
        let info = [
            vk::CalibratedTimestampInfoEXT::default().time_domain(vk::TimeDomainEXT::DEVICE),
            vk::CalibratedTimestampInfoEXT::default().time_domain(self.domain),
        ];
        let mut best = None;
        for _ in 0..3 {
            if let Ok((ticks, deviation)) = unsafe { self.loader.get_calibrated_timestamps(&info) }
            {
                let host_ns = timeline::calibrated_host_ns(ticks[1]);
                let sample = timeline::ClockSample {
                    host_ns,
                    device_tick: ticks[0],
                    period_ns: period,
                    deviation_ns: deviation,
                };
                if best
                    .is_none_or(|previous: timeline::ClockSample| deviation < previous.deviation_ns)
                {
                    best = Some(sample);
                }
            }
        }
        if let Some(sample) = best {
            timeline::record_clock(sample);
        }
    }
}

pub(crate) struct Query {
    device: ash::Device,
    pools: Arc<Mutex<Vec<vk::QueryPool>>>,
    pool: vk::QueryPool,
    cmd: vk::CommandBuffer,
    pub(crate) id: u64,
    ctx: Context,
    queue: &'static str,
    labels: Vec<(&'static str, Context)>,
    host_submit_ns: u64,
    period: f32,
    bits: u32,
    wait: u64,
    signal: u64,
    bytes: u64,
    dispatches: usize,
}

impl Query {
    pub(crate) fn new(
        shared: &VulkanShared,
        cmd: vk::CommandBuffer,
        queue: &'static str,
        bits: u32,
    ) -> crate::Result<Option<Self>> {
        if !timeline::device_enabled() {
            return Ok(None);
        }
        let Some(ctx) = timeline::context() else {
            return Ok(None);
        };
        let _span = timeline::span("trace_query_acquire");
        if bits == 0 {
            return Err(be("timeline queue does not expose timestamps"));
        }
        let clock = shared.timeline_clock.as_ref().ok_or_else(|| {
            be("timeline requires VK_EXT_calibrated_timestamps and a compatible host clock")
        })?;
        clock.sample(shared.submit_timestamp_period_ns, ctx.step);
        let pool = match shared.timeline_query_pools.lock().unwrap().pop() {
            Some(pool) => pool,
            None => unsafe {
                shared.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(QUERY_COUNT),
                    None,
                )
            }
            .map_err(|error| be(format!("create timeline query pool: {error}")))?,
        };
        unsafe {
            shared
                .device
                .cmd_reset_query_pool(cmd, pool, 0, QUERY_COUNT);
            shared
                .device
                .cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, pool, 0);
        }
        Ok(Some(Self {
            device: shared.device.clone(),
            pools: Arc::clone(&shared.timeline_query_pools),
            pool,
            cmd,
            id: timeline::next_submit(),
            ctx,
            queue,
            labels: Vec::with_capacity(32),
            host_submit_ns: 0,
            period: shared.submit_timestamp_period_ns,
            bits,
            wait: 0,
            signal: 0,
            bytes: 0,
            dispatches: 0,
        }))
    }

    pub(crate) fn mark(&mut self, name: &'static str) {
        let name = timeline::device_phase().unwrap_or(name);
        let ctx = timeline::context().unwrap_or(self.ctx);
        if self.queue != "main"
            || self.labels.last().is_some_and(|&(label, previous)| {
                label == name && previous.layer == ctx.layer && previous.step == ctx.step
            })
        {
            return;
        }
        let _span = timeline::span("trace_marker_encode");
        if self.labels.len() >= QUERY_COUNT as usize - 2 {
            return;
        }
        unsafe {
            self.device.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                self.labels.len() as u32 + 1,
            );
        }
        self.labels.push((name, ctx));
    }

    pub(crate) fn close(&mut self, dispatches: usize, wait: u64, signal: u64, bytes: u64) {
        self.dispatches = dispatches;
        self.wait = wait;
        self.signal = signal;
        self.bytes = bytes;
        unsafe {
            self.device.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                self.labels.len() as u32 + 1,
            );
        }
    }

    pub(crate) fn submitted(&mut self) {
        self.host_submit_ns = timeline::clock_ns();
    }

    pub(crate) fn resolve(mut self) {
        let _span = timeline::span("trace_query_readback");
        let mut ticks = vec![0u64; self.labels.len() + 2];
        match unsafe {
            self.device.get_query_pool_results(
                self.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64,
            )
        } {
            Ok(()) => {
                self.event(
                    if self.queue == "main" {
                        "queue_envelope"
                    } else {
                        "DMA_copy"
                    },
                    ticks[0],
                    *ticks.last().unwrap(),
                    self.ctx,
                );
                for (index, &(label, ctx)) in self.labels.iter().enumerate() {
                    self.event(label, ticks[index + 1], ticks[index + 2], ctx);
                }
                self.pools.lock().unwrap().push(self.pool);
                self.pool = vk::QueryPool::null();
            }
            Err(error) => tracing::warn!("completed timeline query was unavailable: {error}"),
        }
    }

    fn event(&self, name: &'static str, start: u64, end: u64, context: Context) {
        timeline::record_device(DeviceEvent {
            name,
            queue: self.queue,
            context,
            submit: self.id,
            host_submit_ns: self.host_submit_ns,
            start_tick: start,
            end_tick: end,
            valid_bits: self.bits,
            period_ns: self.period,
            transfer_wait: self.wait,
            transfer_signal: self.signal,
            bytes: self.bytes,
            dispatches: self.dispatches,
        });
    }
}

impl Drop for Query {
    fn drop(&mut self) {
        if self.pool != vk::QueryPool::null() {
            unsafe {
                self.device.destroy_query_pool(self.pool, None);
            }
        }
    }
}

pub(crate) fn group(kernel: &'static str, config: &infr_core::config::ProfCfg) -> &'static str {
    if config.timeline_raw_kernels {
        kernel
    } else if kernel.contains("qsa") || kernel.contains("attention") || kernel.contains("attn") {
        "QSA_completion_interval"
    } else if kernel.contains("hc_") {
        "HC_completion_interval"
    } else if kernel.contains("deltanet") || kernel.contains("conv1d") {
        "GDN_completion_interval"
    } else if kernel.contains("expert_")
        || kernel.contains("mmv_id")
        || kernel.contains("native_idm")
        || kernel.contains("ple_gate")
        || kernel.contains("moe")
    {
        "Expert_completion_interval"
    } else if kernel.contains("mmv")
        || kernel.contains("gemv")
        || kernel.contains("gemm")
        || kernel.contains("linear")
        || kernel.contains("lin_")
        || kernel.contains("mrow")
    {
        "Projection_completion_interval"
    } else if kernel.contains("norm") || kernel.contains("rope") {
        "NormRope_completion_interval"
    } else if kernel.contains("silu")
        || kernel.contains("sigmoid")
        || kernel.contains("gated")
        || kernel == "add"
        || kernel == "mul"
    {
        "Elementwise_completion_interval"
    } else if kernel.contains("store")
        || kernel.contains("copy")
        || kernel.contains("gather")
        || kernel.contains("scatter")
        || kernel.contains("quant")
        || kernel.contains("zero")
    {
        "Data_completion_interval"
    } else if config.timeline_raw_unknown {
        kernel
    } else {
        "other_completion_interval"
    }
}

#[cfg(test)]
mod tests {
    fn group(kernel: &'static str) -> &'static str {
        super::group(kernel, &Default::default())
    }

    #[test]
    fn raw_kernel_labels_follow_each_backend_config() {
        let mut config = infr_core::config::ProfCfg::default();
        assert_eq!(
            super::group("unknown_test_kernel", &config),
            "other_completion_interval"
        );
        config.timeline_raw_unknown = true;
        assert_eq!(
            super::group("unknown_test_kernel", &config),
            "unknown_test_kernel"
        );
        assert_eq!(
            super::group("expert_gateup", &config),
            "Expert_completion_interval"
        );
        config.timeline_raw_kernels = true;
        assert_eq!(super::group("expert_gateup", &config), "expert_gateup");
        config.timeline_raw_kernels = false;
        config.timeline_raw_unknown = false;
        assert_eq!(
            super::group("expert_gateup", &config),
            "Expert_completion_interval"
        );
    }

    #[test]
    fn prefill_timeline_groups_flashnext_kernels_by_work() {
        assert_eq!(group("expert_gateup"), "Expert_completion_interval");
        assert_eq!(group("expert_down"), "Expert_completion_interval");
        assert_eq!(group("mmv_id_q4_k"), "Expert_completion_interval");
        assert_eq!(
            group("native_idm_iq2s_nr4_paged"),
            "Expert_completion_interval"
        );
        assert_eq!(
            group("native_gemm_warp_q8_0_n128_ag"),
            "Projection_completion_interval"
        );
        assert_eq!(group("native_mrow_q8_0"), "Projection_completion_interval");
        assert_eq!(group("qk_norm_rope_bda"), "NormRope_completion_interval");
        assert_eq!(group("silu_mul"), "Elementwise_completion_interval");
        assert_eq!(group("store_q8_seg"), "Data_completion_interval");
        assert_eq!(group("qsa_indexer_topk"), "QSA_completion_interval");
        assert_eq!(group("attention_q8"), "QSA_completion_interval");
        assert_eq!(group("deltanet_seq"), "GDN_completion_interval");
        assert_eq!(group("qwen_hc_mix"), "HC_completion_interval");
    }
}
