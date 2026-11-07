// Isolate the Vulkan broadcast bias add (Qwen2/2.5 q/k/v `Wx + b`): `dst[r*n+c] = x[r*n+c] +
// bias[c]` for every row. Exercises the shader + recorder directly so a broadcast/indexing bug is
// caught without a full model.
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

#[test]
fn add_bias_broadcast() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // Non-power-of-two, n not a multiple of the 64-wide workgroup — exercises the % and the tail.
    let (rows, n) = (5usize, 7usize);
    let x: Vec<f32> = (0..rows * n).map(|i| i as f32 * 0.1 - 1.0).collect();
    let bias: Vec<f32> = (0..n).map(|c| (c as f32) - 3.0).collect();

    let xbuf = be.alloc(rows * n * 4, BufferUsage::Activations).unwrap();
    be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let bbuf = be.alloc(n * 4, BufferUsage::Activations).unwrap();
    be.upload(bbuf.as_ref(), bytemuck::cast_slice(&bias))
        .unwrap();
    let dst = be.alloc(rows * n * 4, BufferUsage::Activations).unwrap();

    let rec = be.recorder().unwrap();
    rec.add_bias(xbuf.as_ref(), bbuf.as_ref(), dst.as_ref(), rows, n);
    rec.finish().unwrap();

    let mut out = vec![0u8; rows * n * 4];
    be.download(dst.as_ref(), &mut out).unwrap();
    let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out).to_vec();

    for r in 0..rows {
        for c in 0..n {
            let want = x[r * n + c] + bias[c];
            let g = got[r * n + c];
            assert!((g - want).abs() < 1e-6, "r{r} c{c}: got {g}, want {want}");
        }
    }

    // In-place (dst == x, the graph's Qwen2 usage): a second pass adds the bias again.
    let rec = be.recorder().unwrap();
    rec.add_bias(xbuf.as_ref(), bbuf.as_ref(), xbuf.as_ref(), rows, n);
    rec.finish().unwrap();
    let mut out2 = vec![0u8; rows * n * 4];
    be.download(xbuf.as_ref(), &mut out2).unwrap();
    let got2: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out2).to_vec();
    for r in 0..rows {
        for c in 0..n {
            let want = x[r * n + c] + bias[c]; // in place: x already held the raw projection
            assert!(
                (got2[r * n + c] - want).abs() < 1e-6,
                "in-place r{r} c{c}: got {}, want {want}",
                got2[r * n + c]
            );
        }
    }
    eprintln!("Vulkan add_bias broadcast OK ({rows}x{n})");
}
