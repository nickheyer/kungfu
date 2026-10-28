use kf_quant::{Quant, by_name};

const FIXTURES: &[&str] = &[
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_K", "q3_K", "q4_K", "q5_K", "q6_K", "q8_K",
    "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "iq4_nl", "iq3_s", "iq2_s", "iq4_xs", "iq1_m",
    "tq1_0", "tq2_0", "mxfp4", "nvfp4", "q1_0", "q2_0",
];

const LAYOUTS: &[(&str, u32, usize, usize)] = &[
    ("q2_K", 10, 256, 84),
    ("q3_K", 11, 256, 110),
    ("q4_K", 12, 256, 144),
    ("q5_K", 13, 256, 176),
    ("q6_K", 14, 256, 210),
    ("q8_K", 15, 256, 292),
    ("iq2_xxs", 16, 256, 66),
    ("iq2_xs", 17, 256, 74),
    ("iq3_xxs", 18, 256, 98),
    ("iq1_s", 19, 256, 50),
    ("iq4_nl", 20, 32, 18),
    ("iq3_s", 21, 256, 110),
    ("iq2_s", 22, 256, 82),
    ("iq4_xs", 23, 256, 136),
    ("iq1_m", 29, 256, 56),
    ("tq1_0", 34, 256, 54),
    ("tq2_0", 35, 256, 66),
    ("mxfp4", 39, 32, 17),
    ("nvfp4", 40, 64, 36),
    ("q1_0", 41, 128, 18),
    ("q2_0", 42, 64, 18),
];

enum Scale {
    F16(usize),
    F32(usize),
}

const DOT_TYPES: &[(&str, &[Scale])] = &[
    ("q4_K", &[Scale::F16(0), Scale::F16(2)]),
    ("q5_K", &[Scale::F16(0), Scale::F16(2)]),
    ("q6_K", &[Scale::F16(208)]),
    ("q8_K", &[Scale::F32(0)]),
    ("iq4_nl", &[Scale::F16(0)]),
    ("iq4_xs", &[Scale::F16(0)]),
];

struct Fixture {
    n: usize,
    rows: usize,
    raw: Vec<u8>,
    want: Vec<f32>,
}

fn u32_le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Reads one ggml fixture with element count, row count, raw rows and expected f32 output
fn load(name: &str) -> Fixture {
    let path = format!("{}/tests/fixtures/{name}.bin", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = u32_le(&bytes, 0) as usize;
    let rows = u32_le(&bytes, 4) as usize;
    let floats = n * rows;
    let raw_len = bytes.len() - 8 - 4 * floats;
    let raw = bytes[8..8 + raw_len].to_vec();
    let want = (0..floats).map(|i| f32::from_bits(u32_le(&bytes, 8 + raw_len + 4 * i))).collect();
    Fixture { n, rows, raw, want }
}

fn quant(name: &str) -> &'static dyn Quant {
    by_name(name).unwrap_or_else(|| panic!("{name} is not registered"))
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[test]
fn layouts_match_ggml_structs() {
    for &(name, id, block, size) in LAYOUTS {
        let info = quant(name).info();
        assert_eq!((info.id, info.block, info.size), (id, block, size), "{name}");
    }
}

#[test]
fn fixture_rows_match_type_size() {
    for name in FIXTURES {
        let fx = load(name);
        assert_eq!(fx.raw.len(), fx.rows * quant(name).info().row_bytes(fx.n), "{name}");
    }
}

#[test]
fn dequant_is_bit_exact_with_ggml() {
    let mut bad = Vec::new();
    for name in FIXTURES {
        let q = quant(name);
        let fx = load(name);
        let row_bytes = q.info().row_bytes(fx.n);
        let mut got = vec![0.0f32; fx.n];
        let mut mismatches = 0;
        let mut first = None;
        for r in 0..fx.rows {
            q.dequant_row(&fx.raw[r * row_bytes..(r + 1) * row_bytes], &mut got);
            for (i, (g, w)) in got.iter().zip(&fx.want[r * fx.n..(r + 1) * fx.n]).enumerate() {
                if g.to_bits() != w.to_bits() {
                    mismatches += 1;
                    first.get_or_insert((r, i, *g, *w));
                }
            }
        }
        if let Some((r, i, g, w)) = first {
            bad.push(format!("{name}: {mismatches} mismatches, first row {r} index {i} got {g:e} ({:#010x}) want {w:e} ({:#010x})", g.to_bits(), w.to_bits()));
        }
    }
    assert!(bad.is_empty(), "\n{}", bad.join("\n"));
}

#[test]
fn dot_row_matches_dequant_then_dot() {
    let mut rng = Rng(0x6b66_646f_7472_6f77);
    for (name, scales) in DOT_TYPES {
        let q = quant(name);
        let info = q.info();
        let n = 4 * info.block;
        for row in 0..3 {
            let mut src: Vec<u8> = (0..info.row_bytes(n)).map(|_| rng.next() as u8).collect();
            for blk in src.chunks_exact_mut(info.size) {
                for s in *scales {
                    match *s {
                        Scale::F16(at) => {
                            let bits = (((5 + rng.next() % 8) << 10) | (rng.next() & 0x3ff)) as u16;
                            blk[at..at + 2].copy_from_slice(&bits.to_le_bytes());
                        }
                        Scale::F32(at) => {
                            let v = 0.001 + 0.01 * rng.unit();
                            blk[at..at + 4].copy_from_slice(&v.to_le_bytes());
                        }
                    }
                }
            }
            let x: Vec<f32> = (0..n).map(|_| 2.0 * rng.unit() - 1.0).collect();
            let mut deq = vec![0.0f32; n];
            q.dequant_row(&src, &mut deq);
            let want: f64 = deq.iter().zip(&x).map(|(&a, &b)| a as f64 * b as f64).sum();
            let got = q.dot_row(&src, &x) as f64;
            let rel = (got - want).abs() / want.abs();
            assert!(
                rel <= 1e-3,
                "{name} row {row}: dot_row {got} dequant dot {want} relative error {rel}"
            );
        }
    }
}
