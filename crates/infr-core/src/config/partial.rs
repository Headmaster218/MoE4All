//! The partial-layer machinery: `cfg_struct!`, which generates BOTH halves of every config
//! section (the resolved struct in [`super`] and its `Partial*` twin here), plus the value grammar
//! (`ConfigValue`) the dotted-path setter needs.
//!
//! Why one macro for both halves: a `PartialConfig` is the resolved `Config` with every leaf
//! wrapped in `Option` — "this layer specified it" vs "it didn't". Writing the two by hand means
//! 174 chances for them to drift, and a drifted partial silently drops a knob (exactly the failure
//! mode §8.8 exists to catch). The macro also emits the dotted-path accessors, so the TOML schema,
//! `--set`'s path grammar and the tests' field probes are all THE SAME table by construction.
//!
//! The generated items land in [`super`] (that is where the macro is invoked), so `PartialConfig`
//! and the per-section partials are re-exported from there; this module owns their behaviour.
//!
//! Fold rule (§2): a layer only overrides a field it actually specifies. `merge` therefore keeps
//! `self`'s value whenever `other`'s is `None` — a `None` never clobbers a `Some`.

use std::path::PathBuf;

use crate::{DType, SizeSpec};

/// Why a dotted path could not be set. The caller (`--set`, the file layer) turns this into a
/// user-facing message, because only it knows the FULL path (the sections see suffixes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetPathError {
    /// No such field. `--set` makes this a hard error with a did-you-mean; the file layer warns
    /// and ignores (§11 [DECIDE-5]).
    Unknown,
    /// The path exists but the value does not parse into the field's type.
    Value(String),
}

/// A config leaf's value grammar: how `--set`/the TOML file spell it, and how it prints back.
///
/// This is NOT the env grammar — env spellings (`INFR_NO_*` inversion, presence, `flag_from`) live
/// in [`super::env`], because they are per-KEY, not per-TYPE.
pub trait ConfigValue: Sized {
    /// Parse a `--set` / TOML scalar into the field's type.
    fn parse_set(raw: &str) -> Result<Self, String>;
    /// Render for `get_path` (diagnostics + the polarity tests).
    fn to_display(&self) -> String;
}

macro_rules! impl_config_value_fromstr {
    ($($t:ty),* $(,)?) => {$(
        impl ConfigValue for $t {
            fn parse_set(raw: &str) -> Result<Self, String> {
                raw.trim()
                    .parse::<$t>()
                    .map_err(|e| format!("expected a {}: {e}", stringify!($t)))
            }
            fn to_display(&self) -> String {
                self.to_string()
            }
        }
    )*};
}

impl_config_value_fromstr!(usize, u32, u64, f32);

impl ConfigValue for bool {
    fn parse_set(raw: &str) -> Result<Self, String> {
        match raw.trim() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            other => Err(format!("expected a boolean (got {other:?})")),
        }
    }
    fn to_display(&self) -> String {
        self.to_string()
    }
}

impl ConfigValue for String {
    fn parse_set(raw: &str) -> Result<Self, String> {
        Ok(raw.to_string())
    }
    fn to_display(&self) -> String {
        self.clone()
    }
}

impl ConfigValue for PathBuf {
    fn parse_set(raw: &str) -> Result<Self, String> {
        Ok(PathBuf::from(raw))
    }
    fn to_display(&self) -> String {
        self.display().to_string()
    }
}

impl ConfigValue for SizeSpec {
    fn parse_set(raw: &str) -> Result<Self, String> {
        crate::parse_size(raw.trim())
            .ok_or_else(|| format!("expected a size/count like 8192, 256k or 50% (got {raw:?})"))
    }
    fn to_display(&self) -> String {
        match self {
            SizeSpec::Bytes(b) => b.to_string(),
            SizeSpec::Percent(f) => format!("{}%", f * 100.0),
        }
    }
}

impl ConfigValue for DType {
    /// The ONE KV-format name table (`budget::parse_kv_dtype`) — the same spellings the
    /// `INFR_KV_TYPE_K` / `_V` env grammar accepts, so a file entry and an env value cannot mean
    /// different formats.
    fn parse_set(raw: &str) -> Result<Self, String> {
        crate::budget::parse_kv_dtype(raw.trim())
            .ok_or_else(|| format!("not a KV cache format name: {raw:?}"))
    }
    fn to_display(&self) -> String {
        format!("{self:?}")
    }
}

impl ConfigValue for super::MetalDeviceTime {
    /// Named modes (`off` / `flush` / `counters`), not an integer level — the predecessor
    /// (`INFR_METAL_PROFILE`) compared against the exact literals `"2"` and `"3"`, so `=20` read
    /// as "on, level unrecognized" rather than "≥ 2". An unparseable value is an error here (the
    /// file/`--set` layers are strict for known keys), which is what makes that class impossible.
    fn parse_set(raw: &str) -> Result<Self, String> {
        raw.parse()
            .map_err(|()| format!("expected off, flush or counters (got {raw:?})"))
    }
    fn to_display(&self) -> String {
        self.to_string()
    }
}

impl ConfigValue for Vec<usize> {
    /// The multi-GPU device-list grammar — [`super::parse_device_spec`], the ONE copy (S4 folded
    /// `infr-llama`'s duplicate into it). The per-knob minimum device count is NOT checked here
    /// (`min: 0`): that is policy and belongs to the consumer (R5); the env layer applies it
    /// because it must reproduce today's loud error.
    fn parse_set(raw: &str) -> Result<Self, String> {
        super::parse_device_spec(raw, 0)
    }
    fn to_display(&self) -> String {
        self.iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }
}

impl<T: ConfigValue> ConfigValue for Option<T> {
    fn parse_set(raw: &str) -> Result<Self, String> {
        if raw.is_empty() || raw.eq_ignore_ascii_case("none") {
            Ok(None)
        } else {
            T::parse_set(raw).map(Some)
        }
    }
    fn to_display(&self) -> String {
        match self {
            None => "none".to_string(),
            Some(v) => v.to_display(),
        }
    }
}

/// Split `a.b.c` into `("a", Some("b.c"))`; a bare `a` into `("a", None)`.
pub(crate) fn split_path(path: &str) -> (&str, Option<&str>) {
    match path.split_once('.') {
        Some((head, tail)) => (head, Some(tail)),
        None => (path, None),
    }
}

/// Declare one config section: the resolved struct, its `Default` (= today's shipped behaviour),
/// its partial twin, the fold, and the dotted-path accessors.
///
/// Two forms. A section with nested sub-sections spells both groups:
///
/// ```text
/// cfg_struct! {
///     KernelCfg / PartialKernelCfg {
///         subs   { vulkan: VulkanCfg => PartialVulkanCfg, }
///         leaves { qkv_fuse: bool = true, }
///     }
/// }
/// ```
///
/// A leaf-only section writes just the fields (the braced groups are what keep the two
/// repetitions unambiguous — a doc comment can start either kind of entry, so they cannot simply
/// sit next to each other):
///
/// ```text
/// cfg_struct! { ServeCfg / PartialServeCfg { max_tokens_cap: u32 = 131_072, } }
/// ```
macro_rules! cfg_struct {
    (
        $(#[$smeta:meta])*
        $Name:ident / $Partial:ident {
            subs {
                $(
                    $(#[$submeta:meta])*
                    $sf:ident : $ST:ty => $SP:ty,
                )*
            }
            leaves {
                $(
                    $(#[$lmeta:meta])*
                    $lf:ident : $LT:ty = $ldef:expr,
                )*
            }
        }
    ) => {
        $(#[$smeta])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $Name {
            $( $(#[$submeta])* pub $sf: $ST, )*
            $( $(#[$lmeta])* pub $lf: $LT, )*
        }

        impl Default for $Name {
            fn default() -> Self {
                Self {
                    $( $sf: <$ST as Default>::default(), )*
                    $( $lf: $ldef, )*
                }
            }
        }

        impl $Name {
            /// Read one dotted path as a display string. `None` = no such path.
            pub fn get_path(&self, path: &str) -> Option<String> {
                let (head, tail) = $crate::config::partial::split_path(path);
                $(
                    if head == stringify!($sf) {
                        return tail.and_then(|t| self.$sf.get_path(t));
                    }
                )*
                $(
                    if head == stringify!($lf) && tail.is_none() {
                        return Some($crate::config::partial::ConfigValue::to_display(&self.$lf));
                    }
                )*
                None
            }

            /// Append every dotted path under `prefix` (the TOML schema AND `--set`'s path set).
            pub fn collect_paths(prefix: &str, out: &mut Vec<String>) {
                $( <$ST>::collect_paths(&format!("{prefix}{}.", stringify!($sf)), out); )*
                $( out.push(format!("{prefix}{}", stringify!($lf))); )*
            }
        }

        /// Partial twin of the section: every leaf `Option`-wrapped, `None` = "this layer did not
        /// specify it". Generated by `cfg_struct!` — see [`crate::config::partial`].
        #[derive(Debug, Clone, Default, PartialEq)]
        pub struct $Partial {
            $( pub $sf: $SP, )*
            $( pub $lf: ::core::option::Option<$LT>, )*
        }

        impl $Partial {
            /// Fold `other` (a HIGHER-precedence layer) over `self`. A `None` never clobbers.
            pub fn merge(&mut self, other: Self) {
                $( self.$sf.merge(other.$sf); )*
                $( if other.$lf.is_some() { self.$lf = other.$lf; } )*
            }

            /// Write whatever this layer specified onto an already-defaulted section.
            pub fn apply(self, base: &mut $Name) {
                $( self.$sf.apply(&mut base.$sf); )*
                $( if let ::core::option::Option::Some(v) = self.$lf { base.$lf = v; } )*
            }

            /// Set one dotted path from its `--set` / TOML spelling.
            pub fn set_path(
                &mut self,
                path: &str,
                raw: &str,
            ) -> ::core::result::Result<(), $crate::config::SetPathError> {
                let (head, tail) = $crate::config::partial::split_path(path);
                $(
                    if head == stringify!($sf) {
                        return match tail {
                            ::core::option::Option::Some(t) => self.$sf.set_path(t, raw),
                            ::core::option::Option::None => {
                                ::core::result::Result::Err($crate::config::SetPathError::Unknown)
                            }
                        };
                    }
                )*
                $(
                    if head == stringify!($lf) && tail.is_none() {
                        let v = <$LT as $crate::config::partial::ConfigValue>::parse_set(raw)
                            .map_err($crate::config::SetPathError::Value)?;
                        self.$lf = ::core::option::Option::Some(v);
                        return ::core::result::Result::Ok(());
                    }
                )*
                ::core::result::Result::Err($crate::config::SetPathError::Unknown)
            }

            /// Did THIS layer specify `path`? (`--set` vs a bespoke CLI flag, §11 [DECIDE-3].)
            pub fn is_path_set(&self, path: &str) -> bool {
                let (head, tail) = $crate::config::partial::split_path(path);
                $(
                    if head == stringify!($sf) {
                        return tail.is_some_and(|t| self.$sf.is_path_set(t));
                    }
                )*
                $(
                    if head == stringify!($lf) && tail.is_none() {
                        return self.$lf.is_some();
                    }
                )*
                false
            }

            /// Does this layer specify anything at all (recursively)?
            pub fn is_empty(&self) -> bool {
                true $( && self.$sf.is_empty() )* $( && self.$lf.is_none() )*
            }
        }
    };

    // Leaf-only shorthand: forward to the full form with an empty `subs` group.
    (
        $(#[$smeta:meta])*
        $Name:ident / $Partial:ident { $($leaves:tt)* }
    ) => {
        cfg_struct! {
            $(#[$smeta])*
            $Name / $Partial {
                subs {}
                leaves { $($leaves)* }
            }
        }
    };
}

pub(crate) use cfg_struct;
