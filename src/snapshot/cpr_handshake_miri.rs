// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Miri checks for the CPR phase handshake in the parent module.
//!
//! Both tests drive the real `CPRSnapShotMgr`; nothing here is a re-modelling of
//! it. They are compiled only under `cfg(miri)`, so a normal `cargo test` never
//! sees them.
//!
//!   cargo +nightly miri test --lib snapshot::cpr_handshake_miri
//!
//! The manager and a worker each store to their own location and then load the
//! other's, all under `Release`/`Acquire`:
//!
//!   Worker (`reserve_thread_slot`)         Manager
//!     STORE thread_local_states[tid]         STORE global_state / pause_snapshot
//!     LOAD  global_state / pause_snapshot    LOAD  thread_local_states[..]
//!
//! `Release`/`Acquire` orders each thread's own accesses but does not forbid a
//! store from being reordered past a later load of a *different* location, so
//! both loads can miss both stores. The manager then concludes the phase is
//! complete while the worker is still live in the old one, which is exactly the
//! interleaving the comment in `reserve_thread_slot` says the double-check
//! prevents.
//!
//! This is specifically an x86-64 problem. `Release`/`Acquire` lowers to
//! `stlr`/`ldar` on ARM64, and ARMv8's RCsc semantics forbid that reordering, so
//! ARM64 happens to be safe. On x86-64 both are plain `mov`, and store-then-load
//! is the one reordering TSO permits.

use super::*;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;

/// Miri samples weak-memory behaviours rather than enumerating them, so each
/// litmus runs repeatedly. Both violations normally surface almost immediately.
const ITERATIONS: usize = 300;

/// A worker can commit to phase `x` after the manager has both advanced to
/// `x + 1` and satisfied itself that every thread is already there.
///
/// The manager's next act is to run the `x + 1` action against a tree that still
/// has a live writer in phase `x`.
#[test]
fn phase_can_complete_while_a_worker_is_still_in_the_previous_phase() {
    for iteration in 0..ITERATIONS {
        let mgr = Arc::new(CPRSnapShotMgr::new(0));
        let old_state = mgr.global_state.load(Ordering::Relaxed);

        let worker_mgr = mgr.clone();
        let worker = thread::spawn(move || worker_mgr.reserve_thread_slot().ok());

        let new_state = mgr.advance_global_state();
        let phase_completed = mgr.check_if_phase_completed(new_state);

        let Some((tid, version, phase_id)) = worker.join().unwrap() else {
            continue;
        };

        mgr.release_thread_slot(tid);

        let committed = CPRSnapShotMgr::new_snapshot_state(phase_id.as_raw(), version);
        assert!(
            !(phase_completed && committed == old_state),
            "iteration {iteration}: worker committed to the old state {old_state:#x}, \
             but the manager advanced to {new_state:#x} and reported the phase complete"
        );
    }
}

/// The freeze at the top of `sweep` has the same shape, and there the manager's
/// next act is to take exclusive ownership of the tree structure.
///
/// `sweep` collects `Vec<(*const InnerNode, usize)>` from a tree it believes no
/// writer can reach. The `Box` below stands in for one of those inner nodes:
/// reclaiming it while a worker still holds a slot is the use-after-free the
/// freeze is supposed to rule out, and Miri reports the worker's dereference
/// directly.
#[test]
fn sweep_can_reclaim_a_node_a_worker_is_still_allowed_to_reach() {
    struct InnerNodeStandIn {
        magic: u64,
    }

    const MAGIC: u64 = 0x5AFE_0000_5AFE;

    for _ in 0..ITERATIONS {
        let mgr = Arc::new(CPRSnapShotMgr::new(0));
        let node: *mut InnerNodeStandIn =
            Box::into_raw(Box::new(InnerNodeStandIn { magic: MAGIC }));
        let node_addr = node as usize;

        let worker_mgr = mgr.clone();
        let worker = thread::spawn(move || {
            let Ok((tid, _, _)) = worker_mgr.reserve_thread_slot() else {
                return;
            };

            // Holding a slot means the tree is not frozen, so by the protocol
            // this node is live. Miri faults on the next line when it is not.
            let n = unsafe { &*(node_addr as *const InnerNodeStandIn) };
            assert_eq!(n.magic, MAGIC);

            worker_mgr.release_thread_slot(tid);
        });

        // `sweep`: block all reservations, then drain the thread table.
        mgr.pause_snapshot.store(true, Ordering::Release);
        let frozen = mgr.check_if_phase_completed(INVALID_SNAPSHOT_STATE);
        if frozen {
            drop(unsafe { Box::from_raw(node) });
        }

        worker.join().unwrap();

        if !frozen {
            drop(unsafe { Box::from_raw(node) });
        }
    }
}
