//! Dump GGUF metadata + tensor directory. `cargo run -p infr-gguf --example dump -- <path>`
use infr_core::WeightSource;
use infr_gguf::Gguf;

fn main() -> infr_core::Result<()> {
    let path = std::env::args().nth(1).expect("usage: dump <file.gguf>");
    let g = Gguf::open(std::path::Path::new(&path))?;
    println!("=== metadata ===");
    let mut keys: Vec<_> = g.metadata().kv.keys().cloned().collect();
    keys.sort();
    for k in keys {
        let v = g.metadata().get(&k).unwrap();
        let s = format!("{v:?}");
        let s = if s.len() > 110 {
            format!("{}…", &s[..110])
        } else {
            s
        };
        println!("{k} = {s}");
    }
    println!("\n=== tensors ({}) ===", g.tensors().len());
    for t in g.tensors().iter().take(30) {
        println!("{:40} {:?} {:?}", t.name, t.dtype, t.shape);
    }
    Ok(())
}
