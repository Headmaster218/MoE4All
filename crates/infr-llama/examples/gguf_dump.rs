//! Dump GGUF metadata + a sample of tensors. cargo run -p infr-llama --example gguf_dump -- <gguf>
use infr_core::WeightSource;
use infr_gguf::Gguf;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: gguf_dump <gguf>");
    let g = Gguf::open(std::path::Path::new(&path)).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("=== metadata ===");
    let mut keys: Vec<&String> = g.metadata().kv.keys().collect();
    keys.sort();
    for k in keys {
        let s = format!("{:?}", g.metadata().kv[k]);
        let s = if s.len() > 90 {
            format!("{}…", &s[..90])
        } else {
            s
        };
        println!("{k} = {s}");
    }
    println!("\n=== tensors (layer 0 + non-blk) ===");
    for t in g.tensors() {
        if t.name.starts_with("blk.") && !t.name.starts_with("blk.0.") {
            continue;
        }
        println!("{:40} {:?} shape={:?}", t.name, t.dtype, t.shape);
    }
    Ok(())
}
