//! CPU-only preservation of the original planar-Q8/LUT and f64 boundary contracts.
use super::*;
const PERIOD: usize = 64;

fn code(row: usize, head: usize, dim: usize, seed: usize, value: bool) -> i8 {
    let dim = if value { dim % 16 } else { dim };
    ((row % PERIOD * 17 + head * 23 + dim * 11 + seed * 19 + usize::from(value) * 37) % 256) as u8
        as i8
}

fn scale(row: usize, head: usize, dim: usize, seed: usize, value: bool) -> half::f16 {
    let bits = if value {
        0x2400
    } else {
        [0x2000, 0x2400, 0x2800][(row % PERIOD + head + dim / 32 + seed) % 3]
    };
    half::f16::from_bits(bits)
}

fn dequant(row: usize, head: usize, dim: usize, seed: usize, value: bool) -> f64 {
    // Match q8read4h's HALF multiplication, not planar-Q8's unrelated FP32 decode.
    half::f16::from_f32(
        scale(row, head, dim, seed, value).to_f32() * code(row, head, dim, seed, value) as f32,
    )
    .to_f32() as f64
}

fn planar(len: usize, seed: usize, value: bool) -> Vec<u8> {
    let cap = len * NKV * HD;
    let mut data = vec![0u8; cap + cap / 32 * 2];
    for row in 0..len {
        for head in 0..NKV {
            for dim in 0..HD {
                data[(row * NKV + head) * HD + dim] = code(row, head, dim, seed, value) as u8;
            }
            for block in 0..HD / 32 {
                let at = cap + ((row * NKV + head) * HD / 32 + block) * 2;
                data[at..at + 2].copy_from_slice(
                    &scale(row, head, block * 32, seed, value)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
        }
    }
    data
}

fn queries(nh: usize, seed: usize) -> Vec<half::f16> {
    let mut q = vec![half::f16::ZERO; nh * HD];
    for h in 0..nh {
        for (d, value) in [(0, 0.25), (33, -0.125), (97, 0.0625), (225, 0.1875)] {
            q[h * HD + d] = half::f16::from_f32(value * (1 + (h + seed) % 3) as f32);
        }
    }
    q
}

struct FloatReference {
    scores: Vec<[f64; PERIOD]>,
    seed: usize,
    ratio: usize,
}

impl FloatReference {
    fn from_query(query: &[half::f16], seed: usize, nh: usize) -> Self {
        let mut scores = vec![[0.0; PERIOD]; nh];
        for (h, row_scores) in scores.iter_mut().enumerate() {
            for (row, score) in row_scores.iter_mut().enumerate() {
                for dim in [0, 33, 97, 225] {
                    let q = query[h * HD + dim].to_f32() as f64;
                    *score += q * dequant(row, h / (nh / NKV), dim, seed, false);
                }
                *score /= 16.0;
            }
        }
        Self {
            scores,
            seed,
            ratio: nh / NKV,
        }
    }
    fn partial(&self, h: usize, begin: usize, end: usize) -> (f64, f64, [f64; 16]) {
        let mut counts = [0usize; PERIOD];
        for (r, count) in counts.iter_mut().enumerate() {
            let first = begin + (r + PERIOD - begin % PERIOD) % PERIOD;
            if first < end {
                *count = (end - 1 - first) / PERIOD + 1;
            }
        }
        let m = (0..PERIOD)
            .filter(|&r| counts[r] != 0)
            .map(|r| self.scores[h][r])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        let mut values = [0.0; 16];
        for (r, count) in counts.into_iter().enumerate() {
            if count == 0 {
                continue;
            }
            let p = (self.scores[h][r] - m).exp() * count as f64;
            sum += p;
            for (d, acc) in values.iter_mut().enumerate() {
                *acc += p * dequant(r, h / self.ratio, d, self.seed, true);
            }
        }
        (m, sum, values)
    }
}

fn segment_seed(lane_seed: usize, segment: usize) -> usize {
    lane_seed + segment * 13
}

fn segment_bytes(rows: usize, valid_rows: usize, seed: usize, value: bool) -> Vec<u8> {
    assert!(valid_rows <= rows);
    let cap = rows * NKV * HD;
    // A partial last segment still has a FULL physical segment's code/scale layout.
    // Poison unused rows/scales so accidentally including them cannot look correct.
    let mut bytes = vec![0x7fu8; cap + cap / 32 * 2];
    for scale in bytes[cap..].chunks_exact_mut(2) {
        scale.copy_from_slice(&0x7e00u16.to_le_bytes());
    }
    for row in 0..valid_rows {
        for head in 0..NKV {
            for dim in 0..HD {
                bytes[(row * NKV + head) * HD + dim] = code(row, head, dim, seed, value) as u8;
            }
            for block in 0..HD / 32 {
                let at = cap + ((row * NKV + head) * HD / 32 + block) * 2;
                bytes[at..at + 2].copy_from_slice(
                    &scale(row, head, block * 32, seed, value)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
        }
    }
    bytes
}

struct SegReference {
    parts: Vec<FloatReference>,
}

impl SegReference {
    fn new(query: &[half::f16], seed: usize, nh: usize, len: usize) -> Self {
        Self {
            parts: (0..len.div_ceil(SEG_ROWS))
                .map(|seg| FloatReference::from_query(query, segment_seed(seed, seg), nh))
                .collect(),
        }
    }
    fn partial(&self, h: usize, begin: usize, end: usize) -> (f64, f64, [f64; 16]) {
        assert!(begin < end);
        let mut pieces = Vec::new();
        for seg in begin / SEG_ROWS..=(end - 1) / SEG_ROWS {
            let start = begin.max(seg * SEG_ROWS) - seg * SEG_ROWS;
            let stop = end.min((seg + 1) * SEG_ROWS) - seg * SEG_ROWS;
            pieces.push(self.parts[seg].partial(h, start, stop));
        }
        let max = pieces.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        let mut values = [0.0; 16];
        for (m, l, pv) in pieces {
            let weight = (m - max).exp();
            sum += l * weight;
            for d in 0..16 {
                values[d] += pv[d] * weight;
            }
        }
        (max, sum, values)
    }
}

#[test]
fn gqa_fixture_host_reference_contract() {
    for value in [false, true] {
        let bytes = planar(129, 2, value);
        let cap = 129 * NKV * HD;
        assert_eq!(bytes.len(), cap + cap / 32 * 2);
        for row in [0usize, 63, 64, 128] {
            for head in 0..NKV {
                for dim in 0..HD {
                    let at = (row * NKV + head) * HD + dim;
                    let scale_at = cap + (at / 32) * 2;
                    let d = half::f16::from_bits(u16::from_le_bytes([
                        bytes[scale_at],
                        bytes[scale_at + 1],
                    ]));
                    let actual = half::f16::from_f32(d.to_f32() * (bytes[at] as i8) as f32);
                    assert_eq!(actual.to_f32() as f64, dequant(row, head, dim, 2, value));
                }
            }
        }
    }
    for nh in [16, 32] {
        let q = queries(nh, 2);
        let reference = FloatReference::from_query(&q, 2, nh);
        for h in [0, nh / 2, nh - 1] {
            for (begin, end) in [(0, 1), (3, 65), (64, 129), (27, 1025)] {
                let (m, l, pv) = reference.partial(h, begin, end);
                let mut direct_l = 0.0;
                let mut direct_pv = [0.0; 16];
                for row in begin..end {
                    let score: f64 = (0..HD)
                        .map(|d| {
                            q[h * HD + d].to_f32() as f64
                                * dequant(row, h / (nh / NKV), d, 2, false)
                        })
                        .sum::<f64>()
                        / 16.0;
                    let p = (score - m).exp();
                    direct_l += p;
                    for d in 0..16 {
                        direct_pv[d] += p * dequant(row, h / (nh / NKV), d, 2, true);
                    }
                }
                assert!((direct_l - l).abs() < 1e-9);
                for d in 0..16 {
                    assert!((direct_pv[d] - pv[d]).abs() < 1e-9);
                }
            }
        }
    }
}

#[test]
fn segmented_fixture_host_contract() {
    assert_eq!(VALUES_PER_SEGMENT, 1usize << SEG_SHIFT);
    assert_eq!(SEG_SHIFT, 24);
    for value in [false, true] {
        let bytes = segment_bytes(8, 5, 7, value);
        let cap = 8 * NKV * HD;
        assert_eq!(bytes.len(), cap + cap / 32 * 2);
        for row in 0..5 {
            for head in 0..NKV {
                for dim in 0..HD {
                    let elem = (row * NKV + head) * HD + dim;
                    let at = cap + elem / 32 * 2;
                    let d = half::f16::from_bits(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
                    let actual = half::f16::from_f32(d.to_f32() * (bytes[elem] as i8) as f32);
                    assert_eq!(actual.to_f32() as f64, dequant(row, head, dim, 7, value));
                }
            }
        }
        assert!(bytes[5 * NKV * HD..cap].iter().all(|v| *v == 0x7f));
        assert!(bytes[cap + 5 * NKV * HD / 32 * 2..]
            .chunks_exact(2)
            .all(|v| u16::from_le_bytes([v[0], v[1]]) == 0x7e00));
    }
    let nh = 16;
    let q = queries(nh, 1);
    let reference = SegReference::new(&q, 1, nh, 150_000);
    for (begin, end) in [(32760, 32776), (65530, 65540), (149_990, 150_000)] {
        let (m, l, pv) = reference.partial(9, begin, end);
        let mut direct_l = 0.0;
        let mut direct_pv = [0.0; 16];
        for row in begin..end {
            let seg = row / SEG_ROWS;
            let local = row % SEG_ROWS;
            let seed = segment_seed(1, seg);
            let score = (0..HD)
                .map(|d| q[9 * HD + d].to_f32() as f64 * dequant(local, 1, d, seed, false))
                .sum::<f64>()
                / 16.0;
            let weight = (score - m).exp();
            direct_l += weight;
            for d in 0..16 {
                direct_pv[d] += weight * dequant(local, 1, d, seed, true);
            }
        }
        assert!((direct_l - l).abs() < 1e-10);
        for d in 0..16 {
            assert!((direct_pv[d] - pv[d]).abs() < 1e-10);
        }
    }
    for len in [30_000usize, 32767, 32768, 32769, 150_000] {
        for row in [0, len - 1] {
            for d in [0, 255, 256, 511] {
                let elem = row * NKV * HD + d;
                assert_eq!(elem >> SEG_SHIFT, row / SEG_ROWS);
                assert_eq!(
                    elem & (VALUES_PER_SEGMENT - 1),
                    (row % SEG_ROWS) * NKV * HD + d
                );
            }
        }
    }
}
