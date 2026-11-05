//! Parameters omitted by lovemefan's SenseVoiceSmall GGUF converter.
//!
//! This is an explicit compatibility profile, not a generic ASR-model default.
//! Sources: FunAudioLLM/SenseVoice model.py (LayerNorm, lid_dict, encoder), and
//! FunAudioLLM/SenseVoiceSmall config.yaml + am.mvn at revision
//! 3847d57b6bdf2dd8875cb1508d2af43d80a16bf7 on Hugging Face.
//! am.mvn contains these same 80 coefficients repeated seven times after LFR.

use anyhow::{bail, ensure, Context, Result};
use infr_core::{MetaValue, Metadata};

use crate::frontend::FrontendConfig;

#[derive(Clone, Debug)]
pub struct SenseVoiceConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub linear_units: usize,
    /// Includes the one 560 -> 512 encoders0 layer.
    pub encoder_layers: usize,
    pub tp_layers: usize,
    pub kernel_size: usize,
    pub pad_left: usize,
    pub pad_right: usize,
    pub norm_epsilon: f32,
    pub blank_id: usize,
    pub vocab_size: usize,
    /// Names every absent field filled by the documented official Small profile.
    pub compatibility_defaults: Vec<&'static str>,
}

pub(super) struct Parameters {
    pub config: SenseVoiceConfig,
    pub frontend: FrontendConfig,
    pub tokens: Vec<String>,
}

pub(super) fn parse(md: &Metadata) -> Result<Parameters> {
    ensure!(
        md.str("general.architecture") == Some("SenseVoiceSmall"),
        "expected general.architecture=SenseVoiceSmall"
    );
    // Only the known public Small profile is supported in v1. Missing required
    // counts, mistyped fields and conflicting values are errors, not defaults.
    for (key, expected) in [
        ("frontend.sample_rate", 16000),
        ("frontend.num_mels", 80),
        ("frontend.lfr_m", 7),
        ("frontend.lfr_n", 6),
        ("encoder.output_size", 512),
        ("encoder.attention_heads", 4),
        ("encoder.linear_units", 2048),
        ("encoder.num_blocks", 50),
        ("encoder.tp_blocks", 20),
        ("encoder.kernel_size", 11),
        ("encoder.sanm_shfit", 0),
        ("tokenizer.vocab_size", 25055),
    ] {
        expect_count(md, key, expected)?;
    }
    ensure!(
        md.str("frontend.window") == Some("hamming"),
        "SenseVoiceSmall requires frontend.window=hamming"
    );
    expect_number(md, "frontend.frame_length", 25.0)?;
    expect_number(md, "frontend.frame_shift", 10.0)?;
    for (key, expected) in [
        ("encoder.input_size", 560),
        ("model.input_size", 560),
        ("model.blank_id", 0),
        ("tokenizer.ggml.padding_token_id", 0),
    ] {
        if md.get(key).is_some() {
            expect_count(md, key, expected)?;
        }
    }
    for (key, expected) in [
        ("encoder.normalize_before", true),
        ("encoder.concat_after", false),
    ] {
        if let Some(value) = md.get(key) {
            ensure!(
                value == &MetaValue::Bool(expected),
                "unsupported {key}={value:?}"
            );
        }
    }
    let tokens = md
        .get("tokenizer.ggml.tokens")
        .and_then(MetaValue::as_arr)
        .context("missing tokenizer.ggml.tokens array")?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .context("tokenizer.ggml.tokens must contain only strings")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        tokens.len() == 25055,
        "tokenizer token count disagrees with vocab_size"
    );
    // These ids distinguish the public model's blank convention and prompt table
    // from other models that happen to have the same dimensions.
    for (id, piece) in [
        (0, "<unk>"),
        (1, "<s>"),
        (2, "</s>"),
        (24884, "<|zh|>"),
        (24885, "<|en|>"),
        (24888, "<|yue|>"),
        (24892, "<|ja|>"),
        (24896, "<|ko|>"),
        (24992, "<|nospeech|>"),
        (25001, "<|HAPPY|>"),
        (25004, "<|NEUTRAL|>"),
        (25009, "<|EMO_UNKNOWN|>"),
        (25016, "<|withitn|>"),
        (25017, "<|woitn|>"),
    ] {
        ensure!(
            tokens[id] == piece,
            "unsupported SenseVoiceSmall vocabulary: token {id} must be {piece}"
        );
    }
    let mut defaults = Vec::new();
    // nn.LayerNorm(in_size), without eps override, is 1e-5. Do not use
    // sense-voice.cpp's historical 1e-6 setting.
    let mut has_epsilon = false;
    for key in [
        "encoder.layer_norm_epsilon",
        "encoder.layer_norm_eps",
        "encoder.norm_eps",
    ] {
        if md.get(key).is_some() {
            expect_number(md, key, 1e-5)?;
            has_epsilon = true;
        }
    }
    if !has_epsilon {
        defaults.push("encoder.layer_norm_epsilon=1e-5 (official nn.LayerNorm)");
    }
    if md.get("model.blank_id").is_none() {
        defaults.push("model.blank_id=0 (official CTC; token 0 is <unk>)");
    }
    let (cmvn_add, cmvn_mul) = match (md.get("frontend.cmvn_add"), md.get("frontend.cmvn_mul")) {
        (Some(add), Some(mul)) => (
            cmvn_array(add, "frontend.cmvn_add", false)?,
            cmvn_array(mul, "frontend.cmvn_mul", true)?,
        ),
        (None, None) => {
            defaults.push("frontend.cmvn_add/cmvn_mul=official SenseVoiceSmall am.mvn");
            (CMVN_ADD.repeat(7), CMVN_MUL.repeat(7))
        }
        _ => bail!("CMVN requires both frontend.cmvn_add and frontend.cmvn_mul"),
    };
    Ok(Parameters {
        config: SenseVoiceConfig {
            input_size: 560,
            hidden_size: 512,
            attention_heads: 4,
            linear_units: 2048,
            encoder_layers: 50,
            tp_layers: 20,
            kernel_size: 11,
            pad_left: 5,
            pad_right: 5,
            norm_epsilon: 1e-5,
            blank_id: 0,
            vocab_size: tokens.len(),
            compatibility_defaults: defaults,
        },
        frontend: FrontendConfig {
            sample_rate: 16000,
            num_mels: 80,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            lfr_m: 7,
            lfr_n: 6,
            cmvn_add,
            cmvn_mul,
        },
        tokens,
    })
}

fn expect_count(md: &Metadata, key: &str, expected: u64) -> Result<()> {
    ensure!(
        md.u64(key) == Some(expected),
        "unsupported/missing {key}; expected integer {expected}"
    );
    Ok(())
}

fn expect_number(md: &Metadata, key: &str, expected: f64) -> Result<()> {
    let value = md
        .get(key)
        .and_then(MetaValue::as_f64)
        .with_context(|| format!("missing numeric {key}"))?;
    ensure!(
        value.is_finite() && (value - expected).abs() <= expected.abs().max(1e-6) * 1e-6,
        "unsupported {key}={value}; expected {expected}"
    );
    Ok(())
}

fn cmvn_array(value: &MetaValue, key: &str, positive: bool) -> Result<Vec<f32>> {
    let array = value
        .as_arr()
        .with_context(|| format!("{key} must be a numeric array"))?;
    ensure!(
        array.len() == 560,
        "{key} must contain 560 values after LFR"
    );
    array
        .iter()
        .map(|v| {
            let x = v
                .as_f64()
                .with_context(|| format!("{key} contains a nonnumeric value"))?
                as f32;
            ensure!(
                x.is_finite() && (!positive || x > 0.0),
                "invalid value in {key}"
            );
            Ok(x)
        })
        .collect()
}

const CMVN_ADD: [f32; 80] = [
    -8.311879, -8.600912, -9.615928, -10.43595, -11.21292, -11.88333, -12.36243, -12.63706,
    -12.8818, -12.83066, -12.89103, -12.95666, -13.19763, -13.40598, -13.49113, -13.5546,
    -13.55639, -13.51915, -13.68284, -13.53289, -13.42107, -13.65519, -13.50713, -13.75251,
    -13.76715, -13.87408, -13.73109, -13.70412, -13.56073, -13.53488, -13.54895, -13.56228,
    -13.59408, -13.62047, -13.64198, -13.66109, -13.62669, -13.58297, -13.57387, -13.4739,
    -13.53063, -13.48348, -13.61047, -13.64716, -13.71546, -13.79184, -13.90614, -14.03098,
    -14.18205, -14.35881, -14.48419, -14.60172, -14.70591, -14.83362, -14.92122, -15.00622,
    -15.05122, -15.03119, -14.99028, -14.92302, -14.86927, -14.82691, -14.7972, -14.76909,
    -14.71356, -14.61277, -14.51696, -14.42252, -14.36405, -14.30451, -14.23161, -14.19851,
    -14.16633, -14.15649, -14.10504, -13.99518, -13.79562, -13.3996, -12.7767, -11.71208,
];
const CMVN_MUL: [f32; 80] = [
    0.155775, 0.154484, 0.1527379, 0.1518718, 0.1506028, 0.1489256, 0.147067, 0.1447061, 0.1436307,
    0.1443568, 0.1451849, 0.1455157, 0.1452821, 0.1445717, 0.1439195, 0.1435867, 0.1436018,
    0.1438781, 0.1442086, 0.1448844, 0.1454756, 0.145663, 0.146268, 0.1467386, 0.1472724, 0.147664,
    0.1480913, 0.1483739, 0.1488841, 0.1493636, 0.1497088, 0.1500379, 0.1502916, 0.1505389,
    0.1506787, 0.1507102, 0.1505992, 0.1505445, 0.1505938, 0.1508133, 0.1509569, 0.1512396,
    0.1514625, 0.1516195, 0.1516156, 0.1515561, 0.1514966, 0.1513976, 0.1512612, 0.151076,
    0.1510596, 0.1510431, 0.151077, 0.1511168, 0.1511917, 0.151023, 0.1508045, 0.1505885,
    0.1503493, 0.1502373, 0.1501726, 0.1500762, 0.1500065, 0.1499782, 0.150057, 0.1502658,
    0.150469, 0.1505335, 0.1505505, 0.1505328, 0.1504275, 0.1502438, 0.1499674, 0.1497118,
    0.1494661, 0.1493102, 0.1493681, 0.1495501, 0.1499738, 0.1509654,
];

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn official_metadata() -> Metadata {
        let mut md = Metadata::default();
        for (k, v) in [
            ("frontend.sample_rate", 16000),
            ("frontend.num_mels", 80),
            ("frontend.lfr_m", 7),
            ("frontend.lfr_n", 6),
            ("encoder.output_size", 512),
            ("encoder.attention_heads", 4),
            ("encoder.linear_units", 2048),
            ("encoder.num_blocks", 50),
            ("encoder.tp_blocks", 20),
            ("encoder.kernel_size", 11),
            ("encoder.sanm_shfit", 0),
            ("tokenizer.vocab_size", 25055),
        ] {
            md.kv.insert(k.into(), MetaValue::I64(v));
        }
        md.kv.insert(
            "general.architecture".into(),
            MetaValue::Str("SenseVoiceSmall".into()),
        );
        md.kv
            .insert("frontend.window".into(), MetaValue::Str("hamming".into()));
        md.kv
            .insert("frontend.frame_length".into(), MetaValue::I64(25));
        md.kv
            .insert("frontend.frame_shift".into(), MetaValue::I64(10));
        let mut tokens = vec![MetaValue::Str("piece".into()); 25055];
        for (i, s) in [
            (0, "<unk>"),
            (1, "<s>"),
            (2, "</s>"),
            (24884, "<|zh|>"),
            (24885, "<|en|>"),
            (24888, "<|yue|>"),
            (24892, "<|ja|>"),
            (24896, "<|ko|>"),
            (24992, "<|nospeech|>"),
            (25001, "<|HAPPY|>"),
            (25004, "<|NEUTRAL|>"),
            (25009, "<|EMO_UNKNOWN|>"),
            (25016, "<|withitn|>"),
            (25017, "<|woitn|>"),
        ] {
            tokens[i] = MetaValue::Str(s.into());
        }
        md.kv
            .insert("tokenizer.ggml.tokens".into(), MetaValue::Arr(tokens));
        md
    }

    #[test]
    fn known_profile_is_explicit_and_cmvn_is_after_lfr() {
        let p = parse(&official_metadata()).unwrap();
        assert_eq!(p.config.compatibility_defaults.len(), 3);
        assert_eq!(p.config.norm_epsilon, 1e-5);
        assert_eq!(p.config.blank_id, 0);
        assert_eq!(p.frontend.cmvn_add.len(), 560);
        for block in p.frontend.cmvn_add.chunks_exact(80) {
            assert_eq!(block, CMVN_ADD);
        }
        for block in p.frontend.cmvn_mul.chunks_exact(80) {
            assert_eq!(block, CMVN_MUL);
        }
    }

    #[test]
    fn rejects_unknown_missing_negative_and_mistyped_metadata() {
        for (key, value) in [
            (
                "general.architecture",
                MetaValue::Str("SenseVoiceLarge".into()),
            ),
            ("encoder.attention_heads", MetaValue::I64(-4)),
            ("encoder.num_blocks", MetaValue::F64(50.0)),
            ("encoder.layer_norm_epsilon", MetaValue::F64(1e-6)),
            ("encoder.normalize_before", MetaValue::Bool(false)),
            ("frontend.frame_shift", MetaValue::F64(f64::NAN)),
        ] {
            let mut md = official_metadata();
            md.kv.insert(key.into(), value);
            assert!(parse(&md).is_err(), "{key}");
        }
        let mut md = official_metadata();
        md.kv.remove("encoder.kernel_size");
        assert!(parse(&md).is_err());
        md = official_metadata();
        if let MetaValue::Arr(tokens) = md.kv.get_mut("tokenizer.ggml.tokens").unwrap() {
            tokens[0] = MetaValue::Str("<blank>".into());
        }
        assert!(
            parse(&md).is_err(),
            "unknown blank convention must not inherit the profile"
        );
    }

    #[test]
    fn cmvn_is_never_partially_defaulted() {
        let mut md = official_metadata();
        md.kv.insert(
            "frontend.cmvn_add".into(),
            MetaValue::Arr(vec![MetaValue::F64(0.0); 560]),
        );
        assert!(parse(&md).is_err());
        md.kv.insert(
            "frontend.cmvn_mul".into(),
            MetaValue::Arr(vec![MetaValue::F64(1.0); 560]),
        );
        let p = parse(&md).unwrap();
        assert_eq!(p.frontend.cmvn_add, vec![0.0; 560]);
        assert_eq!(p.config.compatibility_defaults.len(), 2);
        md.kv.insert(
            "frontend.cmvn_mul".into(),
            MetaValue::Arr(vec![MetaValue::F64(1.0); 80]),
        );
        assert!(parse(&md).is_err());
    }
}
