//! Shared progress-bar style (thin wrapper over `indicatif`) so every long operation — model
//! downloads, weight loading, … — looks identical. Use [`bar`] wherever you need progress.
//!
//! Layout is a single auto-width line: details on the left/right, the bar fills the middle:
//! ```text
//! label  37/40 layers [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━] 3s
//! label  4.2GB/8.4GB  [━━━━━━━━━━━━━━━━╾────────────] 180MB/s ETA 23s
//! ```

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::io::IsTerminal;

/// What the bar counts — controls the left/right stat fields (the layout is identical either way).
pub enum Unit {
    /// Byte transfer: `done/total … rate ETA` (e.g. downloads).
    Bytes,
    /// A count of discrete items named by the string (e.g. `Items("layers")`).
    Items(&'static str),
}

/// A consistently-styled, auto-width progress bar.
///
/// - `total: Some(n)` → a full-width bar; `None` → a spinner (unknown length).
/// - `label` is the left-most field (the `{msg}`).
/// - Hidden when stderr isn't a TTY (piped output, `infr serve`) so it never spams logs.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn bar(total: Option<u64>, label: &str, unit: Unit) -> ProgressBar {
    if !std::io::stderr().is_terminal() {
        return ProgressBar::hidden();
    }
    styled(total, label, unit)
}

/// The shared style + label, with no visibility decision of its own. [`bar`] draws it straight to
/// stderr and hides it off a TTY; [`bar_in`] hands it to a [`group`], whose draw target then
/// decides — which is why the check cannot live in here.
fn styled(total: Option<u64>, label: &str, unit: Unit) -> ProgressBar {
    // Single line: `{msg} <left> [{wide_bar}] <right>` — wide_bar absorbs all free terminal width.
    let (left, right) = match unit {
        Unit::Bytes => (
            "{bytes}/{total_bytes}".to_string(),
            "{bytes_per_sec} ETA {eta}",
        ),
        Unit::Items(name) => (format!("{{pos}}/{{len}} {name}"), "{elapsed}"),
    };
    let pb = match total {
        Some(n) => {
            let pb = ProgressBar::new(n);
            pb.set_style(
                ProgressStyle::with_template(&format!(
                    "{{msg}}  {left} [{{wide_bar:.cyan/blue}}] {right}"
                ))
                .unwrap()
                .progress_chars("━━╾─"),
            );
            pb
        }
        None => {
            // Unknown length → spinner (no bar to fill); same left/right fields.
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template(&format!(
                    "{{msg}}  {{spinner:.green}} {left} {right}"
                ))
                .unwrap(),
            );
            pb
        }
    };
    pb.set_message(label.to_owned());
    pb
}

/// A set of bars that share the terminal, one line each.
///
/// Needed as soon as two things make progress at the same time — `infr pull` fetching several
/// shards of a split model concurrently. Independent [`bar`]s all address the same last line of
/// the terminal and overwrite each other's output; a group gives each of its bars a line and
/// redraws them together.
///
/// Hidden off a TTY, exactly like [`bar`] — and the hiding propagates to every bar added with
/// [`bar_in`], because `MultiProgress::add` REPLACES a bar's own draw target with the group's.
pub fn group() -> MultiProgress {
    if std::io::stderr().is_terminal() {
        MultiProgress::new()
    } else {
        MultiProgress::with_draw_target(ProgressDrawTarget::hidden())
    }
}

/// A [`bar`] that draws as one line of `group`.
pub fn bar_in(group: &MultiProgress, total: Option<u64>, label: &str, unit: Unit) -> ProgressBar {
    group.add(styled(total, label, unit))
}

#[cfg(test)]
mod tests {
    use super::*;
    use indicatif::TermLike;
    use std::sync::Mutex;

    /// A fake terminal that keeps everything written to it, so a test can read what the bars drew.
    /// `cargo test` captures stderr and `Term::is_term()` is false there, which would make every
    /// real draw target hidden and every assertion below vacuous.
    #[derive(Debug, Default, Clone)]
    struct FakeTerm {
        written: std::sync::Arc<Mutex<String>>,
    }

    impl FakeTerm {
        fn text(&self) -> String {
            self.written.lock().unwrap().clone()
        }
    }

    impl TermLike for FakeTerm {
        fn width(&self) -> u16 {
            120
        }
        fn move_cursor_up(&self, _n: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn move_cursor_down(&self, _n: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn move_cursor_right(&self, _n: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn move_cursor_left(&self, _n: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn write_line(&self, s: &str) -> std::io::Result<()> {
            self.written.lock().unwrap().push_str(s);
            self.written.lock().unwrap().push('\n');
            Ok(())
        }
        fn write_str(&self, s: &str) -> std::io::Result<()> {
            self.written.lock().unwrap().push_str(s);
            Ok(())
        }
        fn clear_line(&self) -> std::io::Result<()> {
            Ok(())
        }
        fn flush(&self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Concurrent bars must each keep their own line rather than overwrite one another — the whole
    /// reason [`group`] exists. Both labels present in one draw is what "each got a line" looks
    /// like from out here.
    #[test]
    fn a_group_draws_every_member() {
        let term = FakeTerm::default();
        let g =
            MultiProgress::with_draw_target(ProgressDrawTarget::term_like(Box::new(term.clone())));
        let a = bar_in(&g, Some(10), "shard-one", Unit::Bytes);
        let b = bar_in(&g, Some(10), "shard-two", Unit::Bytes);
        a.inc(1);
        b.inc(1);
        let drawn = term.text();
        assert!(drawn.contains("shard-one"), "{drawn:?}");
        assert!(drawn.contains("shard-two"), "{drawn:?}");
    }

    /// The GROUP decides whether its bars draw. `MultiProgress::add` replaces a bar's own draw
    /// target, so off a TTY the hiding has to come from the group or `infr pull` sprays escape
    /// sequences into a piped log — and a bar that never joined the group would report hidden here
    /// for the wrong reason (stderr is captured under `cargo test`), which is what the visible half
    /// rules out.
    #[test]
    fn group_visibility_decides_its_bars() {
        let visible = MultiProgress::with_draw_target(ProgressDrawTarget::term_like(Box::new(
            FakeTerm::default(),
        )));
        assert!(!bar_in(&visible, Some(10), "x", Unit::Bytes).is_hidden());
        let hidden = MultiProgress::with_draw_target(ProgressDrawTarget::hidden());
        assert!(bar_in(&hidden, Some(10), "x", Unit::Bytes).is_hidden());
    }
}
