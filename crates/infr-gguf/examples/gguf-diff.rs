//! Diff the tensor catalogs of the original Quality GGUF and the grafted MTPFIX GGUF:
//! same tensor name set? same dtypes/shapes? Plus a numeric sanity probe of key tensors
//! (a broken offset/requant shows up as extreme or denormal-dominated values).
use infr_core::loader::WeightSource;
use infr_gguf::Gguf;
use std::collections::BTreeMap;

fn catalog(path: &std::path::Path) -> BTreeMap<String, (String, Vec<usize>, u64, usize)> {
    let g = Gguf::open(path).expect("open gguf");
    let mut map = BTreeMap::new();
    for t in g.tensors() {
        map.insert(
            t.name.clone(),
            (
                format!("{:?}", t.dtype),
                t.shape.clone(),
                t.offset,
                t.nbytes,
            ),
        );
    }
    map
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: gguf-diff <original.gguf> <grafted.gguf>");
        std::process::exit(1);
    }
    let pa = std::path::PathBuf::from(&args[1]);
    let pb = std::path::PathBuf::from(&args[2]);
    let a = catalog(&pa);
    let b = catalog(&pb);

    let (mut only_a, mut only_b, mut changed) = (0usize, 0usize, 0usize);
    println!("== tensors only in ORIGINAL ==");
    for (name, v) in &a {
        if !b.contains_key(name) {
            println!("  - {name} {} {:?}", v.0, v.1);
            only_a += 1;
        }
    }
    println!("== tensors only in GRAFTED ==");
    for (name, v) in &b {
        if !a.contains_key(name) {
            println!("  + {name} {} {:?}", v.0, v.1);
            only_b += 1;
        }
    }
    println!("== changed dtype/shape/bytes ==");
    for (name, va) in &a {
        if let Some(vb) = b.get(name) {
            if va != vb {
                println!(
                    "  ~ {name}: {} {:?} {}B  ->  {} {:?} {}B",
                    va.0, va.1, va.3, vb.0, vb.1, vb.3
                );
                changed += 1;
            }
        }
    }
    println!(
        "summary: original={} grafted={} only_original={} only_grafted={} changed={}",
        a.len(),
        b.len(),
        only_a,
        only_b,
        changed
    );

    let probes = [
        "blk.0.ffn_gate_exps.weight",
        "blk.23.ffn_down_exps.weight",
        "token_embd.weight",
        "output.weight",
        "blk.39.attn_q.weight",
        "blk.40.ffn_gate_exps.weight",
        "blk.40.attn_q.weight",
        "blk.40.nextn.eh_proj.weight",
    ];
    println!("== numeric probe (mean/std of first 1024 values) ==");
    for path in [&pa, &pb] {
        println!("  -- {}", path.display());
        let g = Gguf::open(&path).expect("open gguf");
        for p in probes {
            let Some(info) = g.tensors().iter().find(|t| t.name == *p) else {
                continue;
            };
            let bytes = match g.tensor_bytes(&info.name) {
                Ok(b) => b,
                Err(e) => {
                    println!("     {p}: read error {e}");
                    continue;
                }
            };
            match infr_gguf::dequant::dequant_block(info.dtype, bytes) {
                Ok(vals) => {
                    let n = vals.len().min(1024);
                    let mean = vals[..n].iter().sum::<f32>() / n as f32;
                    let var = vals[..n]
                        .iter()
                        .map(|v| (v - mean) * (v - mean))
                        .sum::<f32>()
                        / n as f32;
                    println!(
                        "     {p}: mean={mean:.5} std={:.5} min={:.4} max={:.4}",
                        var.sqrt(),
                        vals[..n].iter().cloned().fold(f32::INFINITY, f32::min),
                        vals[..n].iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    );
                }
                Err(e) => println!("     {p}: dequant error {e}"),
            }
        }
    }
}
