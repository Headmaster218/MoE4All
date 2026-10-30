//! Pure layout for ordered resident weights. No Vulkan objects or device activity.

use infr_core::error::{backend as be, Result};

pub(crate) const ADDRESSING_UNIT_MAX: u64 = 1 << 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WeightPlacement {
    pub block: usize,
    pub offset: u64,
    pub requested_bytes: usize,
    pub padded_bytes: u64,
}

#[derive(Debug, Default)]
pub(crate) struct WeightBlockLayout {
    /// Exact used capacity, including each tensor's alignment padding (no block-size floor).
    pub capacity_bytes: u64,
    pub requested_bytes: u64,
}

#[derive(Debug)]
pub(crate) struct WeightLayout {
    pub blocks: Vec<WeightBlockLayout>,
    pub placements: Vec<WeightPlacement>,
}

fn padded_size(size: u64, alignment: u64) -> Result<u64> {
    size.max(1)
        .checked_add(alignment - 1)
        .map(|n| n / alignment * alignment)
        .ok_or_else(|| be("resident weight layout alignment overflow"))
}

impl WeightLayout {
    pub fn build(sizes: &[usize], min_alignment: u64, supported_block_cap: u64) -> Result<Self> {
        let alignment = min_alignment.max(256);
        if !alignment.is_power_of_two() {
            return Err(be("resident weight alignment must be a power of two"));
        }
        let cap = supported_block_cap.min(ADDRESSING_UNIT_MAX);
        let mut blocks: Vec<WeightBlockLayout> = Vec::new();
        let mut placements = Vec::with_capacity(sizes.len());
        for (index, &size) in sizes.iter().enumerate() {
            if size as u64 >= ADDRESSING_UNIT_MAX {
                return Err(be(format!(
                    "resident weight {index} ({size} bytes) must be smaller than 4 GiB"
                )));
            }
            let padded_bytes = padded_size(size as u64, alignment)?;
            if padded_bytes > cap {
                return Err(be(format!(
                    "resident weight {index} ({size} requested, {padded_bytes} padded bytes) \
                     exceeds supported block cap ({cap} bytes)"
                )));
            }
            // Ordered next-fit: never split a tensor, and never reserve an unused block tail.
            if blocks.last().is_none_or(|b| {
                b.capacity_bytes
                    .checked_add(padded_bytes)
                    .is_none_or(|end| end > cap)
            }) {
                blocks.push(WeightBlockLayout::default());
            }
            let block = blocks.len() - 1;
            let b = &mut blocks[block];
            let offset = b.capacity_bytes;
            b.capacity_bytes = offset
                .checked_add(padded_bytes)
                .ok_or_else(|| be("resident weight block capacity overflow"))?;
            b.requested_bytes = b
                .requested_bytes
                .checked_add(size as u64)
                .ok_or_else(|| be("resident weight payload overflow"))?;
            placements.push(WeightPlacement {
                block,
                offset,
                requested_bytes: size,
                padded_bytes,
            });
        }
        Ok(Self { blocks, placements })
    }

    /// Validate before advancing the caller's cursor; a mismatch is recoverable, never fallback.
    pub fn expected(&self, index: usize, size: usize) -> Result<WeightPlacement> {
        let placement = self.placements.get(index).ok_or_else(|| {
            be(format!(
                "unexpected resident weight allocation #{index} ({size} bytes): batch exhausted"
            ))
        })?;
        if size != placement.requested_bytes {
            return Err(be(format!(
                "resident weight allocation #{index}: expected {} bytes, got {size}",
                placement.requested_bytes
            )));
        }
        Ok(*placement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MIB: u64 = 1024 * 1024;

    #[test]
    fn qwen_35b_176_210_mib_weights_share_exact_blocks() {
        // Synthetic 35B-style large projections: eight 176/210 MiB pairs plus small norms.
        // The on-demand allocator would give each large tensor its own exact-size allocation.
        let sizes: Vec<usize> = (0..8)
            .flat_map(|_| [176 * MIB as usize, 4096, 210 * MIB as usize, 4096])
            .collect();
        let layout = WeightLayout::build(&sizes, 256, ADDRESSING_UNIT_MAX).unwrap();
        assert_eq!(layout.blocks.len(), 1);
        let payload = sizes.iter().map(|&s| s as u64).sum::<u64>();
        assert_eq!(layout.blocks[0].capacity_bytes, payload);
        assert_eq!(layout.blocks[0].requested_bytes, payload);
        assert!(payload > 3 * 1024 * MIB);
        assert_eq!(layout.placements[3].offset, 386 * MIB + 4096);

        let full_sizes = sizes.repeat(5); // Forty pairs, well beyond one 4 GiB addressing unit.
        let full = WeightLayout::build(&full_sizes, 256, ADDRESSING_UNIT_MAX).unwrap();
        assert_eq!(full.blocks.len(), 4);
        assert_eq!(
            full.blocks.iter().map(|b| b.capacity_bytes).sum::<u64>(),
            5 * payload
        );
        assert!(full.placements.iter().any(|p| p.offset > (1 << 31)));
        for p in &full.placements {
            assert!(p.offset + p.padded_bytes <= full.blocks[p.block].capacity_bytes);
        }
    }

    #[test]
    fn alignment_multiblock_and_zero_sizes_have_disjoint_reserved_ranges() {
        let layout = WeightLayout::build(&[1, 0, 513, 0], 512, 1536).unwrap();
        assert_eq!(
            layout
                .blocks
                .iter()
                .map(|b| b.capacity_bytes)
                .collect::<Vec<_>>(),
            [1024, 1536]
        );
        assert_eq!(
            layout
                .placements
                .iter()
                .map(|p| (p.block, p.offset, p.padded_bytes))
                .collect::<Vec<_>>(),
            [(0, 0, 512), (0, 512, 512), (1, 0, 1024), (1, 1024, 512)]
        );
        assert_eq!(layout.blocks[0].requested_bytes, 1);
        assert_eq!(layout.blocks[1].requested_bytes, 513);
        assert!(WeightLayout::build(&[], 256, 0).unwrap().blocks.is_empty());
    }

    #[test]
    fn minimum_alignment_and_exact_used_capacity() {
        let layout = WeightLayout::build(&[257, 1], 16, 4096).unwrap();
        assert_eq!(layout.blocks[0].capacity_bytes, 768);
        assert_eq!(layout.blocks[0].requested_bytes, 258);
        assert_eq!(layout.placements[1].offset, 512);
        assert!(WeightLayout::build(&[257], 256, 511).is_err());
        assert!(WeightLayout::build(&[0], 256, 0).is_err());
        assert!(WeightLayout::build(&[1], 768, 4096).is_err());
    }

    #[test]
    fn sequence_mismatch_and_extra_allocation_do_not_match() {
        let layout = WeightLayout::build(&[16, 32], 256, 4096).unwrap();
        assert!(layout.expected(0, 32).is_err());
        assert_eq!(layout.expected(0, 16).unwrap().requested_bytes, 16);
        assert!(layout.expected(1, 16).is_err());
        assert_eq!(layout.expected(1, 32).unwrap().offset, 256);
        assert!(layout.expected(2, 0).is_err());
    }

    #[test]
    fn alignment_arithmetic_overflow_is_an_error() {
        assert!(padded_size(u64::MAX, 256).is_err());
        assert_eq!(padded_size(u64::MAX - 255, 256).unwrap(), u64::MAX - 255);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn four_gib_is_a_block_limit_not_a_total_model_limit() {
        let cap = ADDRESSING_UNIT_MAX;
        let sizes = [(cap - 256) as usize, 256, 512];
        let layout = WeightLayout::build(&sizes, 256, u64::MAX).unwrap();
        assert_eq!(layout.blocks[0].capacity_bytes, cap);
        assert_eq!(layout.blocks[1].capacity_bytes, 512);
        assert_eq!(layout.placements[1].offset, cap - 256);
        assert_eq!(layout.placements[2].offset, 0);
        assert!(WeightLayout::build(&[cap as usize], 256, cap).is_err());
        assert!(WeightLayout::build(&[usize::MAX], 256, u64::MAX).is_err());
        assert!(WeightLayout::build(&[1025], 256, 1024).is_err());
        assert_eq!(
            WeightLayout::build(&[(cap - 1) as usize], 256, cap)
                .unwrap()
                .blocks[0]
                .capacity_bytes,
            cap
        );
    }
}
