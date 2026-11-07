use super::{deliver_token_batch, TokenDelivery};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::time::Duration;

const LANES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Send(usize, u32),
    Ack(usize),
}

struct TraceDelivery {
    sent: [bool; LANES],
    acknowledged: [bool; LANES],
    events: Vec<Event>,
}

impl Default for TraceDelivery {
    fn default() -> Self {
        Self {
            sent: [true; LANES],
            acknowledged: [true; LANES],
            events: Vec::new(),
        }
    }
}

impl TokenDelivery for TraceDelivery {
    fn send_token(&mut self, lane: usize, token: u32) -> bool {
        self.events.push(Event::Send(lane, token));
        self.sent[lane]
    }

    fn receive_ack(&mut self, lane: usize) -> bool {
        self.events.push(Event::Ack(lane));
        self.acknowledged[lane]
    }
}

fn frontier(lanes: usize) -> Vec<(usize, u32)> {
    (0..lanes).map(|lane| (lane, 100 + lane as u32)).collect()
}

#[test]
fn one_to_eight_lanes_deliver_all_tokens_before_any_ack() {
    for lanes in 1..=LANES {
        let tokens = frontier(lanes);
        let mut delivery = TraceDelivery::default();
        let accepted = deliver_token_batch::<LANES>(&tokens, &mut delivery);
        let expected = tokens
            .iter()
            .map(|&(lane, token)| Event::Send(lane, token))
            .chain((0..lanes).map(Event::Ack))
            .collect::<Vec<_>>();
        assert_eq!(delivery.events, expected);
        assert!(accepted[..lanes].iter().all(|&value| value));
        assert!(accepted[lanes..].iter().all(|&value| !value));
    }
}

#[test]
fn false_or_disconnected_ack_never_short_circuits_other_lanes() {
    // Both cancellation and ACK disconnection map to false at the transport boundary.
    for lanes in 1..=LANES {
        for stop_mask in 0..(1usize << lanes) {
            let mut delivery = TraceDelivery::default();
            for lane in 0..lanes {
                delivery.acknowledged[lane] = stop_mask & (1 << lane) == 0;
            }
            let accepted = deliver_token_batch::<LANES>(&frontier(lanes), &mut delivery);
            assert_eq!(delivery.events.len(), 2 * lanes);
            for lane in 0..lanes {
                assert_eq!(accepted[lane], stop_mask & (1 << lane) == 0);
                assert_eq!(delivery.events[lanes + lane], Event::Ack(lane));
            }
        }
    }
}

#[test]
fn failed_send_does_not_wait_for_an_ack_but_other_sends_and_acks_continue() {
    for lanes in 1..=LANES {
        for send_mask in 0..(1usize << lanes) {
            let mut delivery = TraceDelivery::default();
            for lane in 0..lanes {
                delivery.sent[lane] = send_mask & (1 << lane) != 0;
            }
            let accepted = deliver_token_batch::<LANES>(&frontier(lanes), &mut delivery);
            let expected_acks = (0..lanes)
                .filter(|&lane| delivery.sent[lane])
                .map(Event::Ack)
                .collect::<Vec<_>>();
            assert_eq!(&delivery.events[lanes..], expected_acks.as_slice());
            assert_eq!(&accepted[..lanes], &delivery.sent[..lanes]);
        }
    }
}

#[test]
fn sparse_sampled_frontiers_keep_lane_identity_and_skip_omitted_rows() {
    for lanes in 1..=LANES {
        for sample_from in 0..=lanes {
            for eos_mask in 0..(1usize << lanes) {
                let tokens = frontier(lanes)
                    .into_iter()
                    .skip(sample_from)
                    .filter(|&(lane, _)| eos_mask & (1 << lane) == 0)
                    .collect::<Vec<_>>();
                let mut delivery = TraceDelivery::default();
                let accepted = deliver_token_batch::<LANES>(&tokens, &mut delivery);
                assert_eq!(delivery.events.len(), tokens.len() * 2);
                for lane in 0..LANES {
                    assert_eq!(
                        accepted[lane],
                        lane >= sample_from && lane < lanes && eos_mask & (1 << lane) == 0
                    );
                }
            }
        }
    }
}

struct Channels {
    events: SyncSender<u32>,
    acknowledgements: Receiver<bool>,
}

struct ChannelDelivery(Vec<Option<Channels>>);

impl TokenDelivery for ChannelDelivery {
    fn send_token(&mut self, lane: usize, token: u32) -> bool {
        self.0[lane]
            .as_ref()
            .is_some_and(|channels| channels.events.send(token).is_ok())
    }

    fn receive_ack(&mut self, lane: usize) -> bool {
        self.0[lane]
            .as_ref()
            .is_some_and(|channels| channels.acknowledgements.recv().unwrap_or(false))
    }
}

#[test]
fn blocked_first_callback_allows_all_other_rendezvous_receivers_to_start() {
    for lanes in 1..=LANES {
        let (observed_tx, observed_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let mut release_rx = Some(release_rx);
        let mut channels = Vec::new();
        let mut receivers = Vec::new();
        for lane in 0..lanes {
            let (event_tx, event_rx) = mpsc::sync_channel(0);
            let (ack_tx, ack_rx) = mpsc::sync_channel(0);
            let observed_tx = observed_tx.clone();
            let release = (lane == 0).then(|| release_rx.take().unwrap());
            channels.push(Some(Channels {
                events: event_tx,
                acknowledgements: ack_rx,
            }));
            receivers.push(std::thread::spawn(move || {
                let token = event_rx.recv().unwrap();
                observed_tx.send((lane, token)).unwrap();
                if let Some(release) = release {
                    release.recv().unwrap();
                }
                ack_tx.send(lane != 0).unwrap();
            }));
        }
        drop(observed_tx);
        let worker = std::thread::spawn(move || {
            deliver_token_batch::<LANES>(&frontier(lanes), &mut ChannelDelivery(channels))
        });
        let mut observed = Vec::new();
        for _ in 0..lanes {
            match observed_rx.recv_timeout(Duration::from_secs(5)) {
                Ok(event) => observed.push(event),
                Err(_) => break,
            }
        }
        // Release and join before assertions so a regression cannot strand worker threads.
        release_tx.send(()).unwrap();
        let accepted = worker.join().unwrap();
        for receiver in receivers {
            receiver.join().unwrap();
        }
        observed.sort_unstable();
        assert_eq!(observed, frontier(lanes));
        assert!(!accepted[0]);
        assert!(accepted[1..lanes].iter().all(|&value| value));
    }
}

#[test]
fn missing_and_disconnected_channels_do_not_block_healthy_lane() {
    let (dead_event_tx, dead_event_rx) = mpsc::sync_channel(0);
    drop(dead_event_rx);
    let (unused_ack_tx, unused_ack_rx) = mpsc::sync_channel(0);
    let (lost_ack_event_tx, lost_ack_event_rx) = mpsc::sync_channel(0);
    let (lost_ack_tx, lost_ack_rx) = mpsc::sync_channel(0);
    drop(lost_ack_tx);
    let (healthy_event_tx, healthy_event_rx) = mpsc::sync_channel(0);
    let (healthy_ack_tx, healthy_ack_rx) = mpsc::sync_channel(0);
    let receiver = std::thread::spawn(move || {
        assert_eq!(lost_ack_event_rx.recv().unwrap(), 102);
        assert_eq!(healthy_event_rx.recv().unwrap(), 103);
        healthy_ack_tx.send(true).unwrap();
    });
    let mut delivery = ChannelDelivery(vec![
        None,
        Some(Channels {
            events: dead_event_tx,
            acknowledgements: unused_ack_rx,
        }),
        Some(Channels {
            events: lost_ack_event_tx,
            acknowledgements: lost_ack_rx,
        }),
        Some(Channels {
            events: healthy_event_tx,
            acknowledgements: healthy_ack_rx,
        }),
    ]);
    let accepted = deliver_token_batch::<LANES>(&frontier(4), &mut delivery);
    receiver.join().unwrap();
    assert_eq!(
        accepted,
        [false, false, false, true, false, false, false, false]
    );
    drop(unused_ack_tx);
}

#[test]
fn consecutive_frontiers_leave_no_stale_acks_or_tokens() {
    for lanes in 1..=LANES {
        let mut channels = Vec::new();
        let mut receivers = Vec::new();
        for lane in 0..lanes {
            let (event_tx, event_rx) = mpsc::sync_channel(0);
            let (ack_tx, ack_rx) = mpsc::sync_channel(0);
            channels.push(Some(Channels {
                events: event_tx,
                acknowledgements: ack_rx,
            }));
            receivers.push(std::thread::spawn(move || {
                for round in 0..3 {
                    assert_eq!(event_rx.recv().unwrap(), round * 100 + lane as u32);
                    ack_tx.send(round < 2).unwrap();
                }
            }));
        }
        let mut delivery = ChannelDelivery(channels);
        for round in 0..3 {
            let tokens = (0..lanes)
                .map(|lane| (lane, round * 100 + lane as u32))
                .collect::<Vec<_>>();
            let accepted = deliver_token_batch::<LANES>(&tokens, &mut delivery);
            assert!(accepted[..lanes].iter().all(|&value| value == (round < 2)));
        }
        for receiver in receivers {
            receiver.join().unwrap();
        }
    }
}

#[test]
fn receiver_callback_unwind_disconnects_its_ack_and_drains_healthy_lanes() {
    for lanes in 2..=LANES {
        let mut channels = Vec::new();
        let mut receivers = Vec::new();
        for lane in 0..lanes {
            let (event_tx, event_rx) = mpsc::sync_channel(0);
            let (ack_tx, ack_rx) = mpsc::sync_channel(0);
            channels.push(Some(Channels {
                events: event_tx,
                acknowledgements: ack_rx,
            }));
            receivers.push(std::thread::spawn(move || {
                // The request owns BOTH endpoints inside the unwinding scope, just like
                // wait_for_decode_batch. Catching outside it must not retain an ACK sender.
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    let token = event_rx.recv().unwrap();
                    assert_eq!(token, 100 + lane as u32);
                    let on_token = |_| {
                        if lane == 0 {
                            panic!("request output callback failed");
                        }
                    };
                    on_token(token);
                    // A zero-capacity send completes only if the producer actually drains it.
                    ack_tx
                        .send(true)
                        .expect("healthy lane ACK must be consumed");
                }))
                .is_err()
            }));
        }
        let producer = std::thread::spawn(move || {
            deliver_token_batch::<LANES>(&frontier(lanes), &mut ChannelDelivery(channels))
        });
        let result = producer.join();
        let unwound: Vec<_> = receivers.into_iter().map(|thread| thread.join()).collect();
        let accepted = result.unwrap();
        assert!(
            !accepted[0],
            "unwind resolves the successful send as a disconnected ACK"
        );
        assert!(accepted[1..lanes].iter().all(|&ack| ack));
        assert!(accepted[lanes..].iter().all(|&ack| !ack));
        for (lane, outcome) in unwound.into_iter().enumerate() {
            assert_eq!(outcome.unwrap(), lane == 0, "receiver lane={lane}");
        }
    }
}

#[test]
#[should_panic(expected = "duplicate token delivery lane")]
fn duplicate_lane_is_rejected_before_any_channel_io() {
    deliver_token_batch::<LANES>(&[(0, 1), (0, 2)], &mut TraceDelivery::default());
}

#[test]
#[should_panic(expected = "token delivery lane exceeds cohort capacity")]
fn invalid_lane_is_rejected_before_any_channel_io() {
    deliver_token_batch::<LANES>(&[(LANES, 1)], &mut TraceDelivery::default());
}
