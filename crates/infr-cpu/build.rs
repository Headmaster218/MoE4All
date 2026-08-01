// INFR_PROFILE=1 at build time -> cfg(infr_profile) -> #[cfg_attr(infr_profile,
// infr_prof::instrument)] annotations become live and inject profiling spans into every fn.
// Default builds get NO cfg, the attributes vanish, and zero profiling code is compiled in.
// See docs/perf/playbook.md § "Build-time auto-instrumentation".
fn main() {
    println!("cargo:rerun-if-env-changed=INFR_PROFILE");
    println!("cargo:rustc-check-cfg=cfg(infr_profile)");
    if std::env::var("INFR_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0") {
        println!("cargo:rustc-cfg=infr_profile");
    }
}
