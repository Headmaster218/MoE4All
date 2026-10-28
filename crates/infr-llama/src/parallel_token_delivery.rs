//! CPU-only transport for one parallel Decode frontier.

pub(crate) type TokenAcknowledgements = [bool; infr_core::backend::MAX_DECODE_ROWS];

/// Legacy callbacks complete delivery and ACK per token. Ordinary Decode can deliver a
/// complete non-EOS frontier before collecting ACKs. Both must resolve every supplied row.
pub(crate) enum ParallelTokenCallback<'a> {
    Legacy(&'a mut dyn FnMut(usize, u32) -> bool),
    Batch(&'a mut dyn FnMut(&[(usize, u32)]) -> TokenAcknowledgements),
}

pub(crate) trait TokenDelivery {
    /// True only if the receiver accepted this token event.
    fn send_token(&mut self, lane: usize, token: u32) -> bool;
    /// Receive exactly one ACK. A disconnected receiver returns false.
    fn receive_ack(&mut self, lane: usize) -> bool;
}

/// Deliver one sampled frontier before waiting for any lane's stream callback.
/// The caller supplies only non-EOS rows, in sampling order, at most once per lane.
/// False ACKs stop the next forward but must not leave other ACK senders blocked.
pub(crate) fn deliver_token_batch<const LANES: usize>(
    tokens: &[(usize, u32)],
    delivery: &mut impl TokenDelivery,
) -> [bool; LANES] {
    let mut pending = [false; LANES];
    for &(lane, _) in tokens {
        assert!(lane < LANES, "token delivery lane exceeds cohort capacity");
        assert!(!pending[lane], "duplicate token delivery lane");
        pending[lane] = true;
    }
    for &(lane, token) in tokens {
        pending[lane] = delivery.send_token(lane, token);
    }
    let mut accepted = [false; LANES];
    for &(lane, _) in tokens {
        if pending[lane] {
            accepted[lane] = delivery.receive_ack(lane);
        }
    }
    accepted
}

#[cfg(test)]
#[path = "parallel_token_delivery_tests.rs"]
mod tests;
