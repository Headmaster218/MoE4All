//! Persistent conversation-state boundary used by the server's cold KV cache.
//!
//! This module deliberately describes logical session state, not files. The file/catalog policy
//! lives in `crate::session_cache`; this side owns the authoritative list of device buffers whose
//! bytes affect continuation correctness.

use super::kv_frame_pool::FrameId;
use super::segmented_kv::{PlaneKind, SegmentedKvLayout};
use super::weights::{SeamKv, TurnRecurrentCkpt};
use crate::paged_cache::PagedKv;
use crate::Config;
use anyhow::{anyhow, Result};
use infr_core::backend::{Backend, Buffer};
use infr_core::tensor::DType;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SessionStateMeta {
    pub max_ctx: usize,
    pub k_fmt: DType,
    pub v_fmt: DType,
    pub committed_tokens: usize,
    pub cached: Vec<u32>,
    pub checkpoint_tokens: [Option<Vec<u32>>; super::TURN_CHECKPOINT_COUNT],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum SessionBufferKey {
    K(u32),
    V(u32),
    QsaRaw(u32),
    QsaBlock(u32),
    PleState,
    CheckpointK(u8, u32),
    CheckpointV(u8, u32),
    CheckpointPle(u8),
}

pub(crate) struct SessionBuffer<'a> {
    pub key: SessionBufferKey,
    pub buffer: &'a dyn Buffer,
    pub committed_bytes: usize,
}

impl SeamKv {
    pub(crate) fn session_state_meta(&self) -> SessionStateMeta {
        SessionStateMeta {
            max_ctx: self.max_ctx,
            k_fmt: self.k_fmt,
            v_fmt: self.v_fmt,
            committed_tokens: self.segmented_kv.committed_tokens,
            cached: self.cached.clone(),
            checkpoint_tokens: std::array::from_fn(|index| {
                self.turn_recurrent_ckpts[index]
                    .as_ref()
                    .filter(|checkpoint| checkpoint.valid)
                    .map(|checkpoint| checkpoint.tokens.clone())
            }),
        }
    }

    pub(crate) fn can_release_session_state(&self) -> bool {
        self.segmented_kv.enabled
    }

    pub(crate) fn session_state_buffers<'a>(
        &'a self,
        be: &dyn Backend,
    ) -> Result<Vec<SessionBuffer<'a>>> {
        let mut out = Vec::new();
        for (layer, buffer) in self.kbufs.iter().enumerate() {
            push_buffer(
                &mut out,
                be,
                SessionBufferKey::K(layer as u32),
                buffer.as_ref(),
            )?;
        }
        for (layer, buffer) in self.vbufs.iter().enumerate() {
            push_buffer(
                &mut out,
                be,
                SessionBufferKey::V(layer as u32),
                buffer.as_ref(),
            )?;
        }
        for (layer, buffer) in self.qsa_kbufs.iter().enumerate() {
            if let Some(buffer) = buffer.as_deref() {
                push_buffer(&mut out, be, SessionBufferKey::QsaRaw(layer as u32), buffer)?;
            }
        }
        for (layer, buffer) in self.qsa_cbufs.iter().enumerate() {
            if let Some(buffer) = buffer.as_deref() {
                push_buffer(
                    &mut out,
                    be,
                    SessionBufferKey::QsaBlock(layer as u32),
                    buffer,
                )?;
            }
        }
        if let Some(buffer) = self.ple_state_buf.as_deref() {
            push_buffer(&mut out, be, SessionBufferKey::PleState, buffer)?;
        }
        for (checkpoint_index, checkpoint) in self.turn_recurrent_ckpts.iter().enumerate() {
            if let Some(checkpoint) = checkpoint.as_ref().filter(|checkpoint| checkpoint.valid) {
                let checkpoint_index = checkpoint_index as u8;
                for (index, &layer) in checkpoint.layers.iter().enumerate() {
                    push_buffer(
                        &mut out,
                        be,
                        SessionBufferKey::CheckpointK(checkpoint_index, layer as u32),
                        checkpoint.kbufs[index].as_ref(),
                    )?;
                    push_buffer(
                        &mut out,
                        be,
                        SessionBufferKey::CheckpointV(checkpoint_index, layer as u32),
                        checkpoint.vbufs[index].as_ref(),
                    )?;
                }
                if let Some(buffer) = checkpoint.ple_state.as_deref() {
                    push_buffer(
                        &mut out,
                        be,
                        SessionBufferKey::CheckpointPle(checkpoint_index),
                        buffer,
                    )?;
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn prepare_session_restore(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        meta: &SessionStateMeta,
    ) -> Result<()> {
        if !self.segmented_kv.enabled {
            return Err(anyhow!(
                "cold session restore requires a segmented KV allocation"
            ));
        }
        if meta.max_ctx != self.max_ctx || meta.k_fmt != self.k_fmt || meta.v_fmt != self.v_fmt {
            return Err(anyhow!(
                "cold session geometry differs from the resident slot (ctx {} vs {}, K {:?} vs {:?}, V {:?} vs {:?})",
                meta.max_ctx,
                self.max_ctx,
                meta.k_fmt,
                self.k_fmt,
                meta.v_fmt,
                self.v_fmt,
            ));
        }
        if meta.cached.len() > self.max_ctx || meta.committed_tokens > self.max_ctx {
            return Err(anyhow!(
                "cold session depth exceeds the resident slot capacity"
            ));
        }
        self.segmented_kv.ensure_depth(
            be,
            cfg,
            self.max_ctx,
            self.k_fmt,
            self.v_fmt,
            &self.kbufs,
            &self.vbufs,
            &self.qsa_kbufs,
            &self.qsa_cbufs,
            meta.committed_tokens.max(meta.cached.len()),
        )?;
        for checkpoint in self.turn_recurrent_ckpts.iter_mut().flatten() {
            checkpoint.invalidate();
        }
        for (index, tokens) in meta.checkpoint_tokens.iter().enumerate() {
            if let Some(tokens) = tokens.as_deref() {
                TurnRecurrentCkpt::begin(
                    &mut self.turn_recurrent_ckpts[index],
                    be,
                    cfg,
                    &self.kbufs,
                    &self.vbufs,
                    self.ple_state_buf.as_deref(),
                    tokens,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn session_state_buffer(&self, key: SessionBufferKey) -> Option<&dyn Buffer> {
        let layer = |index: u32| usize::try_from(index).ok();
        match key {
            SessionBufferKey::K(index) => self.kbufs.get(layer(index)?).map(Box::as_ref),
            SessionBufferKey::V(index) => self.vbufs.get(layer(index)?).map(Box::as_ref),
            SessionBufferKey::QsaRaw(index) => self.qsa_kbufs.get(layer(index)?)?.as_deref(),
            SessionBufferKey::QsaBlock(index) => self.qsa_cbufs.get(layer(index)?)?.as_deref(),
            SessionBufferKey::PleState => self.ple_state_buf.as_deref(),
            SessionBufferKey::CheckpointK(checkpoint, index) => checkpoint_buffer(
                self.turn_recurrent_ckpts
                    .get(checkpoint as usize)?
                    .as_ref()?,
                index,
                |checkpoint, position| checkpoint.kbufs[position].as_ref(),
            ),
            SessionBufferKey::CheckpointV(checkpoint, index) => checkpoint_buffer(
                self.turn_recurrent_ckpts
                    .get(checkpoint as usize)?
                    .as_ref()?,
                index,
                |checkpoint, position| checkpoint.vbufs[position].as_ref(),
            ),
            SessionBufferKey::CheckpointPle(checkpoint) => self
                .turn_recurrent_ckpts
                .get(checkpoint as usize)?
                .as_ref()?
                .ple_state
                .as_deref(),
        }
    }

    pub(crate) fn session_restore_buffer_specs(
        &self,
        be: &dyn Backend,
    ) -> Result<Vec<(SessionBufferKey, usize)>> {
        let mut specs = self
            .session_state_buffers(be)?
            .into_iter()
            .map(|buffer| (buffer.key, buffer.committed_bytes))
            .collect::<Vec<_>>();
        for (checkpoint_index, checkpoint) in self.turn_recurrent_ckpts.iter().enumerate() {
            if let Some(checkpoint) = checkpoint
                .as_ref()
                .filter(|checkpoint| !checkpoint.valid && !checkpoint.tokens.is_empty())
            {
                let checkpoint_index = checkpoint_index as u8;
                for (index, &layer) in checkpoint.layers.iter().enumerate() {
                    specs.push((
                        SessionBufferKey::CheckpointK(checkpoint_index, layer as u32),
                        be.buffer_committed_bytes(checkpoint.kbufs[index].as_ref())?,
                    ));
                    specs.push((
                        SessionBufferKey::CheckpointV(checkpoint_index, layer as u32),
                        be.buffer_committed_bytes(checkpoint.vbufs[index].as_ref())?,
                    ));
                }
                if let Some(buffer) = checkpoint.ple_state.as_deref() {
                    specs.push((
                        SessionBufferKey::CheckpointPle(checkpoint_index),
                        be.buffer_committed_bytes(buffer)?,
                    ));
                }
            }
        }
        specs.sort_unstable_by_key(|&(key, _)| key);
        Ok(specs)
    }

    pub(crate) fn finish_session_restore(&mut self, meta: SessionStateMeta) -> Result<()> {
        self.cached = meta.cached;
        self.segmented_kv.committed_tokens = meta.committed_tokens;
        for (index, tokens) in meta.checkpoint_tokens.into_iter().enumerate() {
            match (tokens, self.turn_recurrent_ckpts[index].as_mut()) {
                (Some(tokens), Some(checkpoint)) => {
                    checkpoint.tokens = tokens;
                    checkpoint.copied.fill(true);
                    checkpoint.ple_copied = true;
                    checkpoint.valid = true;
                }
                (None, Some(checkpoint)) => checkpoint.invalidate(),
                (None, None) => {}
                _ => {
                    return Err(anyhow!(
                        "cold session checkpoint shape changed during restore"
                    ))
                }
            }
        }
        Ok(())
    }

    /// Release only the per-token physical ranges. Shared model weights and the slot's small fixed
    /// recurrent/IO buffers stay alive, so the same slot can restore another conversation without
    /// reloading the model.
    pub(crate) fn release_session_state(&mut self, be: &dyn Backend, cfg: &Config) -> Result<()> {
        if !self.segmented_kv.enabled {
            return Err(anyhow!(
                "cold session release requires a segmented KV allocation"
            ));
        }
        be.sync()
            .map_err(|error| anyhow!("sync before KV release: {error}"))?;
        let layout = SegmentedKvLayout::for_qwen(
            cfg,
            self.max_ctx,
            self.k_fmt,
            self.v_fmt,
            self.segmented_kv.page_tokens,
        )
        .ok_or_else(|| anyhow!("segmented KV enabled for a non-Qwen session"))?;
        // Publish the logically-empty state before releasing the first physical plane. If a
        // backend fails partway through, the next use will recommit every required plane instead
        // of trusting the stale committed-depth counter and reading a released address.
        self.segmented_kv.committed_tokens = 0;
        self.cached.clear();
        for checkpoint in self.turn_recurrent_ckpts.iter_mut().flatten() {
            checkpoint.invalidate();
        }
        for plane in layout.planes {
            let buffer: &dyn Buffer = match plane.kind {
                PlaneKind::K => self.kbufs[plane.layer].as_ref(),
                PlaneKind::V => self.vbufs[plane.layer].as_ref(),
                PlaneKind::QsaRaw => self.qsa_kbufs[plane.layer]
                    .as_deref()
                    .ok_or_else(|| anyhow!("missing QSA raw cache at layer {}", plane.layer))?,
                PlaneKind::QsaBlock => self.qsa_cbufs[plane.layer]
                    .as_deref()
                    .ok_or_else(|| anyhow!("missing QSA block cache at layer {}", plane.layer))?,
            };
            be.release_segmented_kv(buffer).map_err(|error| {
                anyhow!("release {:?} at layer {}: {error}", plane.kind, plane.layer)
            })?;
        }
        if let Some(paged) = &self.segmented_kv.paged {
            // Paged tier: the plane loop zeroed every table; hand the session's frame owners back
            // so frames the tree still owns survive and orphaned ones return to the corridor.
            let chain = std::mem::take(&mut self.segmented_kv.chain);
            if !chain.is_empty() {
                paged
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .session_release(&chain);
            }
        }
        Ok(())
    }

    /// The slot's plane buffers in `SegmentedKvLayout::planes` order — the SAME order the
    /// corridor's `plane_offsets` were built from, so the plane index is a shared coordinate.
    fn plane_buffers(&self, cfg: &Config) -> Result<Vec<&dyn Buffer>> {
        let layout = SegmentedKvLayout::for_qwen(
            cfg,
            self.max_ctx,
            self.k_fmt,
            self.v_fmt,
            self.segmented_kv.page_tokens,
        )
        .ok_or_else(|| anyhow!("segmented KV enabled for a non-Qwen session"))?;
        layout
            .planes
            .iter()
            .map(|plane| {
                Ok(match plane.kind {
                    PlaneKind::K => self.kbufs[plane.layer].as_ref(),
                    PlaneKind::V => self.vbufs[plane.layer].as_ref(),
                    PlaneKind::QsaRaw => self.qsa_kbufs[plane.layer]
                        .as_deref()
                        .ok_or_else(|| anyhow!("missing QSA raw cache at layer {}", plane.layer))?,
                    PlaneKind::QsaBlock => {
                        self.qsa_cbufs[plane.layer].as_deref().ok_or_else(|| {
                            anyhow!("missing QSA block cache at layer {}", plane.layer)
                        })?
                    }
                })
            })
            .collect()
    }

    /// Inject (or refresh) the engine-wide paged KV tree on this slot. Called on every checkout:
    /// a slot rebuilt by `fork` starts with `paged: None`, and the tree is engine state, not
    /// conversation state.
    pub(crate) fn set_paged(&mut self, kv: std::sync::Arc<std::sync::Mutex<PagedKv>>) {
        self.segmented_kv.paged = Some(kv);
    }

    /// Point this slot's page table at tree-owned `frames` (the shared prefix chain) and adopt
    /// `tokens` as the materialized conversation. ZERO-COPY: no bytes move — the frames already
    /// hold the KV; reuse is `frames.len() * planes` table writes plus one refcount owner each.
    /// The slot must be logically empty (the caller detaches first) and the GPU baton held.
    pub(crate) fn paged_attach(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        kv: &mut PagedKv,
        frames: &[FrameId],
        tokens: &[u32],
    ) -> Result<()> {
        if !self.segmented_kv.enabled || self.segmented_kv.paged.is_none() {
            return Err(anyhow!("paged attach requires a paged segmented KV slot"));
        }
        if !self.segmented_kv.chain.is_empty() || self.segmented_kv.committed_tokens != 0 {
            return Err(anyhow!(
                "paged attach into a slot that still holds {} frames",
                self.segmented_kv.chain.len()
            ));
        }
        let buffers = self.plane_buffers(cfg)?;
        kv.session_attach(frames);
        let mut failure: Option<String> = None;
        for (page, &frame) in frames.iter().enumerate() {
            for (plane, &buffer) in buffers.iter().enumerate() {
                if let Err(error) = be.set_kv_table_entry(buffer, page, kv.plane_addr(frame, plane))
                {
                    failure = Some(format!("page {page} plane {plane}: {error}"));
                    break;
                }
            }
            if failure.is_some() {
                break;
            }
        }
        if let Some(detail) = failure {
            // Roll back as a unit: drop the owners we just gained and zero every table (some planes
            // may hold a half-written page). The slot returns to logically-empty.
            kv.session_release(frames);
            for &buffer in &buffers {
                let _ = be.release_segmented_kv(buffer);
            }
            self.segmented_kv.committed_tokens = 0;
            self.segmented_kv.chain.clear();
            return Err(anyhow!("paged KV table write at {detail}"));
        }
        self.segmented_kv.chain = frames.to_vec();
        self.segmented_kv.committed_tokens =
            (frames.len() * self.segmented_kv.page_tokens).min(self.max_ctx);
        self.cached = tokens.to_vec();
        for checkpoint in self.turn_recurrent_ckpts.iter_mut().flatten() {
            checkpoint.invalidate();
        }
        Ok(())
    }

    /// Publish the slot's conversation into the tree (complete pages only — the sub-page tail is
    /// dropped, matching the reuse rule) and detach the WHOLE chain: state zeroed, tables cleared,
    /// session owners dropped. Returns the published token count (0 detaches without publishing).
    /// On publish `Err` the slot is untouched; a later table-write `Err` leaves the tree owning
    /// what it published (the caller resets the slot). GPU baton held.
    pub(crate) fn paged_publish_detach(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        kv: &mut PagedKv,
    ) -> Result<usize> {
        if !self.segmented_kv.enabled || self.segmented_kv.paged.is_none() {
            return Err(anyhow!("paged detach requires a paged segmented KV slot"));
        }
        let page = self.segmented_kv.page_tokens;
        let full = self.cached.len() / page;
        if full > 0 {
            // Publish FIRST: until `publish` returns the frames are only session-owned, so a
            // failed publish leaves a consistent resident slot. After it, the tree holds its own
            // owners and the detach below can never strand the conversation.
            kv.publish(
                self.cached[..full * page].to_vec(),
                self.segmented_kv.chain[..full].to_vec(),
            )?;
        }
        be.sync()
            .map_err(|error| anyhow!("sync before paged KV detach: {error}"))?;
        self.segmented_kv.committed_tokens = 0;
        self.cached.clear();
        for checkpoint in self.turn_recurrent_ckpts.iter_mut().flatten() {
            checkpoint.invalidate();
        }
        let buffers = self.plane_buffers(cfg)?;
        for &buffer in &buffers {
            be.release_segmented_kv(buffer)
                .map_err(|error| anyhow!("zero paged KV plane table during detach: {error}"))?;
        }
        let chain = std::mem::take(&mut self.segmented_kv.chain);
        kv.session_release(&chain);
        Ok(full * page)
    }
}

impl SessionStateMeta {
    pub(crate) fn continuation_prefix_len(&self, prompt: &[u32]) -> Option<usize> {
        let live_score = common_prefix_len(&self.cached, prompt);
        let live = (live_score > 0
            && (live_score == self.cached.len() || live_score == prompt.len()))
        .then_some(live_score);
        let checkpoint = self
            .checkpoint_tokens
            .iter()
            .flatten()
            .filter_map(|tokens| {
                (!tokens.is_empty() && tokens.len() < prompt.len() && prompt.starts_with(tokens))
                    .then_some(tokens.len())
            })
            .max();
        live.into_iter().chain(checkpoint).max()
    }
}

fn push_buffer<'a>(
    out: &mut Vec<SessionBuffer<'a>>,
    be: &dyn Backend,
    key: SessionBufferKey,
    buffer: &'a dyn Buffer,
) -> Result<()> {
    let committed_bytes = be
        .buffer_committed_bytes(buffer)
        .map_err(|error| anyhow!("inspect {key:?}: {error}"))?;
    if committed_bytes > 0 {
        out.push(SessionBuffer {
            key,
            buffer,
            committed_bytes,
        });
    }
    Ok(())
}

fn checkpoint_buffer<'a>(
    checkpoint: &'a TurnRecurrentCkpt,
    layer: u32,
    select: impl FnOnce(&'a TurnRecurrentCkpt, usize) -> &'a dyn Buffer,
) -> Option<&'a dyn Buffer> {
    let position = checkpoint
        .layers
        .iter()
        .position(|&candidate| candidate == layer as usize)?;
    Some(select(checkpoint, position))
}

fn common_prefix_len(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(cached: &[u32], checkpoints: [Option<&[u32]>; 2]) -> SessionStateMeta {
        SessionStateMeta {
            max_ctx: 1024,
            k_fmt: DType::Q8_0,
            v_fmt: DType::Q8_0,
            committed_tokens: 1024,
            cached: cached.to_vec(),
            checkpoint_tokens: checkpoints.map(|tokens| tokens.map(<[u32]>::to_vec)),
        }
    }

    #[test]
    fn cold_metadata_uses_the_same_live_continuation_rule() {
        let state = meta(&[1, 2, 3], [None, None]);
        assert_eq!(state.continuation_prefix_len(&[1, 2, 3, 4]), Some(3));
        assert_eq!(state.continuation_prefix_len(&[1, 2, 9]), None);
        assert_eq!(state.continuation_prefix_len(&[1, 2]), Some(2));
    }

    #[test]
    fn cold_metadata_can_resume_a_strict_checkpoint_extension() {
        let state = meta(&[1, 9, 9], [Some(&[1]), Some(&[1, 2])]);
        assert_eq!(state.continuation_prefix_len(&[1, 2, 3]), Some(2));
        assert_eq!(state.continuation_prefix_len(&[1, 2]), Some(1));
        assert_eq!(state.continuation_prefix_len(&[9]), None);
    }
}
