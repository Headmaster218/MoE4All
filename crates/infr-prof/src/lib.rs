//! `#[infr_prof::instrument]` — build-time profiling injection (see docs/perf/playbook.md).
//!
//! Applied (always behind `#[cfg_attr(infr_profile, ...)]`) to a `fn`, an `impl` block, or an
//! inline `mod`, it rewrites EVERY function inside to open an `infr_prof_rt` span on entry and
//! close it via RAII on any exit path. Coverage is therefore per-item: annotate an impl block
//! once and every method in it — including ones added later — is profiled for free.
//!
//! Skipped automatically (no span injected):
//! - `const fn` (no `Instant` in const), `async fn` (guard would cross `.await`/threads),
//!   `#[naked]` fns (no prologue allowed), `#[test]`/`#[bench]` fns
//! - `#[inline]`/`#[inline(always)]` fns — declaring a fn inline asserts it is smaller than
//!   the ~50ns probe pair, so probing it would only measure the probe
//! - fns carrying `#[infr_prof::skip]` (or `#[cfg_attr(infr_profile, infr_prof::skip)]`) —
//!   explicit opt-out for measured-too-hot functions
//! - closures (only `fn` items are touched) and macro invocations inside items
//!
//! Handled correctly: generics (the injected `static` site is shared across instantiations),
//! trait impls (site named `<Ty as Trait>::fn`), recursion (depth-aware inclusive time),
//! `?`/`return`/panic (RAII guard), nested fns (each gets its own span).

use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::visit_mut::VisitMut;
use syn::{parse_macro_input, Attribute, Item, Signature};

/// Rewrite every `fn` in the annotated item (fn / impl block / inline mod) to record a span.
#[proc_macro_attribute]
pub fn instrument(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut item = parse_macro_input!(item as Item);
    Instrumenter { scope: Vec::new() }.visit_item_mut(&mut item);
    item.into_token_stream().into()
}

/// Identity attribute: opt a single fn out of an enclosing `#[instrument]`. `instrument`
/// detects it by name; standalone it expands to nothing extra.
#[proc_macro_attribute]
pub fn skip(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

struct Instrumenter {
    /// Lexical scope inside the annotated item: impl-type names and enclosing fn names.
    /// (Inline `mod` names are NOT pushed — `module_path!()` already reflects them.)
    scope: Vec<String>,
}

impl Instrumenter {
    fn site_name(&self, ident: &syn::Ident) -> String {
        if self.scope.is_empty() {
            ident.to_string()
        } else {
            format!("{}::{}", self.scope.join("::"), ident)
        }
    }

    fn inject(&self, sig: &Signature, block: &mut syn::Block) {
        let name = self.site_name(&sig.ident);
        let prelude: syn::Block = syn::parse2(quote! {{
            static __INFR_PROF_SITE: ::infr_prof_rt::Site =
                ::infr_prof_rt::Site::new(::core::module_path!(), #name);
            let __infr_prof_guard = ::infr_prof_rt::enter(&__INFR_PROF_SITE);
        }})
        .expect("infr-prof: prelude parse");
        block.stmts.splice(0..0, prelude.stmts);
    }
}

/// True iff `path`'s segments are exactly `infr_prof :: skip` (a structural match, so a `skip`
/// from some unrelated crate, or the bare words in a doc string, never qualify).
fn is_infr_prof_skip_path(path: &syn::Path) -> bool {
    let segs: Vec<_> = path.segments.iter().collect();
    segs.len() == 2 && segs[0].ident == "infr_prof" && segs[1].ident == "skip"
}

/// True iff `a` opts its fn out of instrumentation: `#[infr_prof::skip]` directly, or
/// `#[cfg_attr(<pred>, infr_prof::skip)]` (cfg_attr is still unexpanded when this macro runs on
/// an enclosing item). Doc comments are matched structurally on the attribute PATH and so are
/// never triggered — a fn whose docs merely mention "infr_prof" or "skip" stays instrumented.
fn is_skip_attr(a: &Attribute) -> bool {
    let path = a.path();
    if path.is_ident("doc") {
        return false;
    }
    if is_infr_prof_skip_path(path) {
        return true;
    }
    if path.is_ident("cfg_attr") {
        if let Ok(nested) = a.parse_args_with(
            syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
        ) {
            // #[cfg_attr(pred, attr, ...)]: the first entry is the cfg predicate, the rest are
            // the attributes applied when it holds — match infr_prof::skip among those.
            return nested
                .iter()
                .skip(1)
                .any(|m| is_infr_prof_skip_path(m.path()));
        }
    }
    false
}

fn should_skip(attrs: &[Attribute], sig: &Signature) -> bool {
    if sig.constness.is_some() || sig.asyncness.is_some() {
        return true;
    }
    attrs.iter().any(|a| {
        let path = a.path();
        if path.is_ident("test") || path.is_ident("bench") || path.is_ident("naked") {
            return true;
        }
        // Any #[inline] fn is treated as a sub-probe-cost leaf: the author already declared it
        // small enough to inline, so a ~50ns enter/exit pair would dominate it (measured:
        // hadd_i32_xmm at 1.7e9 calls/decode-run). Un-inline'd hot leaves opt out explicitly
        // with #[cfg_attr(infr_profile, infr_prof::skip)].
        if path.is_ident("inline") {
            return true;
        }
        is_skip_attr(a)
    })
}

fn type_scope_name(imp: &syn::ItemImpl) -> String {
    let ty = imp.self_ty.to_token_stream().to_string().replace(' ', "");
    match &imp.trait_ {
        Some((_, path, _)) => {
            let tr = path.to_token_stream().to_string().replace(' ', "");
            format!("<{ty} as {tr}>")
        }
        None => ty,
    }
}

/// True iff the `cfg` predicate `m` is satisfied ONLY when `test` is set: bare `test`, or an
/// `all(...)` with `test` among its conjuncts. NOT `not(test)`, `any(test, ...)` (both compile
/// outside test), `feature = "test-*"`, or any other identifier that merely contains "test".
fn cfg_pred_is_test(m: &syn::Meta) -> bool {
    match m {
        syn::Meta::Path(p) => p.is_ident("test"),
        syn::Meta::List(list) if list.path.is_ident("all") => list
            .parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            )
            .map(|inner| inner.iter().any(cfg_pred_is_test))
            .unwrap_or(false),
        _ => false,
    }
}

/// True iff `a` is a `#[cfg(...)]` whose predicate is test-only (see [`cfg_pred_is_test`]).
fn is_cfg_test(a: &Attribute) -> bool {
    a.path().is_ident("cfg")
        && a.parse_args::<syn::Meta>()
            .map(|m| cfg_pred_is_test(&m))
            .unwrap_or(false)
}

impl VisitMut for Instrumenter {
    fn visit_item_fn_mut(&mut self, node: &mut syn::ItemFn) {
        self.scope.push(node.sig.ident.to_string());
        syn::visit_mut::visit_item_fn_mut(self, node); // nested fns first
        self.scope.pop();
        if !should_skip(&node.attrs, &node.sig) {
            self.inject(&node.sig, &mut node.block);
        }
    }

    fn visit_impl_item_fn_mut(&mut self, node: &mut syn::ImplItemFn) {
        self.scope.push(node.sig.ident.to_string());
        syn::visit_mut::visit_impl_item_fn_mut(self, node);
        self.scope.pop();
        if !should_skip(&node.attrs, &node.sig) {
            self.inject(&node.sig, &mut node.block);
        }
    }

    fn visit_item_impl_mut(&mut self, node: &mut syn::ItemImpl) {
        self.scope.push(type_scope_name(node));
        syn::visit_mut::visit_item_impl_mut(self, node);
        self.scope.pop();
    }

    fn visit_item_mod_mut(&mut self, node: &mut syn::ItemMod) {
        // Leave test-only modules (`#[cfg(test)]` / `#[cfg(all(..., test, ...))]`) alone, but
        // still descend into `#[cfg(not(test))]` and `feature = "test-*"` modules — those are
        // real, non-test code whose predicate merely mentions the word "test".
        let is_test_mod = node.attrs.iter().any(is_cfg_test);
        if !is_test_mod {
            syn::visit_mut::visit_item_mod_mut(self, node);
        }
    }

    // Trait *definitions* (default bodies) and closures: deliberately untouched.
    fn visit_item_trait_mut(&mut self, _node: &mut syn::ItemTrait) {}
    fn visit_expr_closure_mut(&mut self, node: &mut syn::ExprClosure) {
        // Recurse: a closure body may contain nested `fn` items worth instrumenting.
        syn::visit_mut::visit_expr_closure_mut(self, node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse `src` (an attribute source, e.g. `#[infr_prof::skip]`) attached to a dummy fn and
    /// return the first attribute.
    fn fn_attr(src: &str) -> Attribute {
        let f: syn::ItemFn = syn::parse_str(&format!("{src}\nfn f() {{}}")).unwrap();
        f.attrs.into_iter().next().unwrap()
    }

    #[test]
    fn skip_attr_matches_path_not_words() {
        // Real opt-outs: the attribute PATH is infr_prof::skip (direct or via cfg_attr).
        assert!(is_skip_attr(&fn_attr("#[infr_prof::skip]")));
        assert!(is_skip_attr(&fn_attr(
            "#[cfg_attr(infr_profile, infr_prof::skip)]"
        )));
        // A doc comment merely mentioning the words is NOT an opt-out.
        assert!(!is_skip_attr(&fn_attr(
            "#[doc = \"see infr_prof to skip hot leaves\"]"
        )));
        // A `skip` from an unrelated crate is NOT ours.
        assert!(!is_skip_attr(&fn_attr("#[skip]")));
        assert!(!is_skip_attr(&fn_attr("#[some::other::skip]")));
    }

    #[test]
    fn should_skip_honors_path_and_signature() {
        // Doc mentioning skip -> still instrumented.
        let f: syn::ItemFn =
            syn::parse_str("#[doc = \"call infr_prof::skip maybe\"]\nfn f() {}").unwrap();
        assert!(!should_skip(&f.attrs, &f.sig));
        // Real #[infr_prof::skip] -> skipped.
        let f: syn::ItemFn = syn::parse_str("#[infr_prof::skip]\nfn f() {}").unwrap();
        assert!(should_skip(&f.attrs, &f.sig));
        // const / async always skipped regardless of attrs.
        let f: syn::ItemFn = syn::parse_str("const fn f() {}").unwrap();
        assert!(should_skip(&f.attrs, &f.sig));
        let f: syn::ItemFn = syn::parse_str("async fn f() {}").unwrap();
        assert!(should_skip(&f.attrs, &f.sig));
    }

    #[test]
    fn cfg_test_module_detection() {
        // Test-only: skipped.
        assert!(is_cfg_test(&fn_attr("#[cfg(test)]")));
        assert!(is_cfg_test(&fn_attr("#[cfg(all(unix, test))]")));
        // Compiles outside test (or is unrelated): NOT skipped.
        assert!(!is_cfg_test(&fn_attr("#[cfg(not(test))]")));
        assert!(!is_cfg_test(&fn_attr("#[cfg(any(test, unix))]")));
        assert!(!is_cfg_test(&fn_attr("#[cfg(feature = \"test-utils\")]")));
        assert!(!is_cfg_test(&fn_attr("#[cfg(unix)]")));
    }
}
