use super::*;
use infr_core::backend::BufferUsage;

#[test]
#[ignore = "requires a Vulkan GPU"]
fn hc_inject_norm_tracks_both_outputs_before_any_consumer() {
    let backend = VulkanBackend::new().unwrap();
    let ne = 2560usize;
    let hc = 4usize;
    let alloc = |elements| {
        backend
            .alloc(elements * 4, BufferUsage::Activations)
            .unwrap()
    };
    let residual = alloc(ne * hc);
    let block = alloc(ne);
    let gate = alloc(hc);
    let gamma = alloc(ne * hc);
    let wide = alloc(ne * hc);
    let normed = alloc(ne * hc);
    assert_ne!(
        Recorder::vk_handle(wide.as_ref()),
        Recorder::vk_handle(normed.as_ref()),
        "independent Vulkan buffers are needed to expose the missing write hazard"
    );
    let recorder = backend.recorder().unwrap();
    recorder.qwen_hc_inject_norm(
        residual.as_ref(),
        block.as_ref(),
        gate.as_ref(),
        gamma.as_ref(),
        wide.as_ref(),
        normed.as_ref(),
        1,
        hc as u32,
        ne as u32,
        1e-6,
    );
    for output in [&wide, &normed] {
        assert!(recorder
            .dirty_writes
            .borrow()
            .contains(&Recorder::vk_handle(output.as_ref())));
    }
    recorder.finish().unwrap();
    for output in [&wide, &normed] {
        let mut values = vec![f32::NAN; ne * hc];
        backend
            .download(output.as_ref(), bytemuck::cast_slice_mut(&mut values))
            .unwrap();
        assert!(values.iter().all(|value| *value == 0.0));
    }
}
