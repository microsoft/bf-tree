// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        thread, Arc, Barrier,
    },
    BfTree, Config, LeafInsertResult, LeafReadResult, ScanReturnField, StorageBackend,
};

type Model = BTreeMap<u8, Vec<u8>>;

#[derive(Clone, Debug, Serialize)]
enum Operation {
    Put(u8, Vec<u8>),
    Delete(u8),
    Read(u8),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
enum Response {
    Ack,
    Value(Option<Vec<u8>>),
}

#[derive(Clone, Debug, Serialize)]
struct Completed {
    invocation: usize,
    response: usize,
    operation: Operation,
    result: Response,
}

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Linearizable(Model),
    Illegal,
    Inconclusive,
}

fn transition(model: &Model, event: &Completed) -> Option<Model> {
    let mut next = model.clone();
    match (&event.operation, &event.result) {
        (Operation::Put(key, value), Response::Ack) => {
            next.insert(*key, value.clone());
        }
        (Operation::Delete(key), Response::Ack) => {
            next.remove(key);
        }
        (Operation::Read(key), Response::Value(value)) if model.get(key) == value.as_ref() => {}
        _ => return None,
    }
    Some(next)
}

fn search(
    history: &[Completed],
    predecessors: &[u64],
    chosen: u64,
    model: &Model,
    budget: &mut usize,
) -> Verdict {
    if chosen.count_ones() as usize == history.len() {
        return Verdict::Linearizable(model.clone());
    }
    if *budget == 0 {
        return Verdict::Inconclusive;
    }
    *budget -= 1;
    for (index, event) in history.iter().enumerate() {
        let bit = 1 << index;
        if chosen & bit != 0 || predecessors[index] & !chosen != 0 {
            continue;
        }
        if let Some(next) = transition(model, event) {
            match search(history, predecessors, chosen | bit, &next, budget) {
                Verdict::Illegal => {}
                other => return other,
            }
        }
    }
    Verdict::Illegal
}

fn check(history: &[Completed], initial: &Model, mut budget: usize) -> Verdict {
    assert!(history.iter().all(|e| e.invocation < e.response));
    let timestamps: BTreeSet<_> = history
        .iter()
        .flat_map(|e| [e.invocation, e.response])
        .collect();
    assert_eq!(timestamps.len(), 2 * history.len());
    if history.len() > 64 {
        return Verdict::Inconclusive;
    }
    let predecessors: Vec<_> = history
        .iter()
        .map(|event| {
            history.iter().enumerate().fold(0, |mask, (index, other)| {
                if other.response < event.invocation {
                    mask | (1 << index)
                } else {
                    mask
                }
            })
        })
        .collect();
    search(history, &predecessors, 0, initial, &mut budget)
}

fn event(invocation: usize, response: usize, operation: Operation, result: Response) -> Completed {
    Completed {
        invocation,
        response,
        operation,
        result,
    }
}

#[test]
fn checker_backtracks_over_overlapping_writes() {
    let history = [
        event(0, 4, Operation::Put(0, vec![1]), Response::Ack),
        event(1, 2, Operation::Put(0, vec![2]), Response::Ack),
        event(3, 5, Operation::Read(0), Response::Value(Some(vec![1]))),
    ];
    assert_eq!(
        check(&history, &Model::new(), 100),
        Verdict::Linearizable(Model::from([(0, vec![1])]))
    );
}

#[test]
fn checker_rejects_stale_invented_and_cyclic_reads() {
    for history in [
        vec![
            event(0, 1, Operation::Put(0, vec![1]), Response::Ack),
            event(2, 3, Operation::Read(0), Response::Value(None)),
        ],
        vec![event(
            0,
            1,
            Operation::Read(0),
            Response::Value(Some(vec![9])),
        )],
        vec![
            event(0, 3, Operation::Put(0, vec![1]), Response::Ack),
            event(4, 7, Operation::Read(1), Response::Value(None)),
            event(1, 5, Operation::Put(1, vec![2]), Response::Ack),
            event(6, 8, Operation::Read(0), Response::Value(None)),
        ],
    ] {
        assert_eq!(check(&history, &Model::new(), 100), Verdict::Illegal);
    }
}

#[test]
fn checker_handles_delete_reinsert_and_initial_state() {
    let operations = [
        (Operation::Delete(0), Response::Ack),
        (Operation::Delete(0), Response::Ack),
        (Operation::Read(0), Response::Value(None)),
        (Operation::Put(0, vec![2]), Response::Ack),
        (Operation::Read(0), Response::Value(Some(vec![2]))),
    ];
    let history: Vec<_> = operations
        .into_iter()
        .enumerate()
        .map(|(i, (op, result))| event(2 * i, 2 * i + 1, op, result))
        .collect();
    assert_eq!(
        check(&history, &Model::from([(0, vec![1])]), 100),
        Verdict::Linearizable(Model::from([(0, vec![2])]))
    );
}

#[test]
fn checker_budget_exhaustion_is_not_a_pass_or_counterexample() {
    let history = [
        event(0, 3, Operation::Put(0, vec![1]), Response::Ack),
        event(1, 2, Operation::Read(0), Response::Value(None)),
    ];
    assert_eq!(check(&history, &Model::new(), 1), Verdict::Inconclusive);
    assert!(matches!(
        check(&history, &Model::new(), 100),
        Verdict::Linearizable(_)
    ));
}

fn key(id: u8) -> [u8; 8] {
    let mut key = [b'k'; 8];
    key[7] = id;
    key
}

fn invoke(tree: &BfTree, operation: &Operation) -> Response {
    match operation {
        Operation::Put(id, value) => {
            assert_eq!(tree.insert(&key(*id), value), LeafInsertResult::Success);
            Response::Ack
        }
        Operation::Delete(id) => {
            tree.delete(&key(*id));
            Response::Ack
        }
        Operation::Read(id) => {
            let mut buffer = vec![0; 1024];
            match tree.read(&key(*id), &mut buffer) {
                LeafReadResult::Found(length) => {
                    assert!(length as usize <= buffer.len());
                    buffer.truncate(length as usize);
                    Response::Value(Some(buffer))
                }
                LeafReadResult::Deleted | LeafReadResult::NotFound => Response::Value(None),
                LeafReadResult::InvalidKey => panic!("valid eight-byte key rejected"),
            }
        }
    }
}

fn tick(clock: &AtomicUsize) -> usize {
    // A single clock orders invocation/response boundaries across all workers.
    clock.fetch_add(1, Ordering::SeqCst)
}

#[derive(Clone, Copy, Debug)]
enum Scenario {
    Overwrite,
    CompetingInserts,
    DeleteReinsert,
    #[cfg(not(feature = "shuttle"))]
    Split,
}

fn scripts(scenario: Scenario) -> [Vec<Operation>; 2] {
    use Operation::{Delete, Put, Read};
    match scenario {
        Scenario::Overwrite => [
            vec![Put(0, vec![1; 32]), Read(0), Put(0, vec![3; 504]), Read(1)],
            vec![Read(0), Put(0, vec![2; 248]), Read(0), Put(1, vec![4; 56])],
        ],
        Scenario::CompetingInserts => [
            vec![Put(0, vec![1; 32]), Put(1, vec![3; 248]), Read(0), Read(1)],
            vec![Put(0, vec![2; 504]), Put(1, vec![4; 56]), Read(1), Read(0)],
        ],
        Scenario::DeleteReinsert => [
            vec![Delete(0), Read(0), Put(0, vec![3; 504]), Read(0)],
            vec![Read(0), Delete(0), Put(0, vec![2; 248]), Read(0)],
        ],
        #[cfg(not(feature = "shuttle"))]
        Scenario::Split => [
            (0..6).map(|id| Put(id, vec![id; 960])).collect(),
            (0..6).map(Read).collect(),
        ],
    }
}

fn run_history(scenario: Scenario) {
    let mut config = Config::default();
    config
        .storage_backend(StorageBackend::Memory)
        .cache_only(false)
        .cb_size_byte(2 * 1024 * 1024)
        .cb_min_record_size(16)
        .cb_max_record_size(1024)
        .cb_max_key_len(8)
        .read_promotion_rate(100);
    let tree = Arc::new(BfTree::with_config(config, None).unwrap());
    let initial = if matches!(scenario, Scenario::Overwrite | Scenario::DeleteReinsert) {
        Model::from([(0, vec![0; 56])])
    } else {
        Model::new()
    };
    for (id, value) in &initial {
        assert_eq!(tree.insert(&key(*id), value), LeafInsertResult::Success);
    }
    let operations = scripts(scenario);
    let keys: BTreeSet<_> = operations
        .iter()
        .flatten()
        .map(|op| match op {
            Operation::Put(id, _) | Operation::Delete(id) | Operation::Read(id) => *id,
        })
        .collect();
    let clock = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(Barrier::new(2));
    let workers: Vec<_> = operations
        .into_iter()
        .map(|operations| {
            let (tree, clock, start) = (tree.clone(), clock.clone(), start.clone());
            thread::spawn(move || {
                operations
                    .into_iter()
                    .enumerate()
                    .map(|(index, operation)| {
                        let invocation = tick(&clock);
                        if index == 0 {
                            start.wait();
                        }
                        thread::yield_now();
                        let result = invoke(&tree, &operation);
                        event(invocation, tick(&clock), operation, result)
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let mut history: Vec<_> = workers
        .into_iter()
        .flat_map(|worker| worker.join().unwrap())
        .collect();
    // Final reads must agree with the same linearization, not independent ones.
    for id in keys {
        let operation = Operation::Read(id);
        let invocation = tick(&clock);
        let result = invoke(&tree, &operation);
        history.push(event(invocation, tick(&clock), operation, result));
    }
    let verdict = check(&history, &initial, 100_000);
    let Verdict::Linearizable(model) = verdict else {
        panic!(
            "{verdict:?}; initial={}; history={}",
            serde_json::to_string(&initial).unwrap(),
            serde_json::to_string(&history).unwrap()
        );
    };
    let mut scan = tree
        .scan_with_count(&key(0), model.len() + 1, ScanReturnField::KeyAndValue)
        .unwrap();
    let mut buffer = [0; 1024];
    for (id, value) in model {
        assert_eq!(scan.next(&mut buffer), Some((8, value.len())));
        assert_eq!(&buffer[..8], &key(id));
        assert_eq!(&buffer[8..8 + value.len()], value);
    }
    assert_eq!(scan.next(&mut buffer), None);
    #[cfg(not(feature = "shuttle"))]
    if matches!(scenario, Scenario::Split) {
        assert!(
            crate::utils::BfsVisitor::new_all_nodes(&tree)
                .filter(|node| matches!(node, crate::utils::NodeInfo::Leaf { .. }))
                .count()
                > 1
        );
    }
}

#[rstest::rstest]
#[case::overwrite(Scenario::Overwrite)]
#[case::competing_inserts(Scenario::CompetingInserts)]
#[case::delete_reinsert(Scenario::DeleteReinsert)]
fn point_histories(#[case] scenario: Scenario) {
    #[cfg(not(feature = "shuttle"))]
    for _ in 0..32 {
        run_history(scenario);
    }
    #[cfg(feature = "shuttle")]
    {
        let run = move || run_history(scenario);
        if let Ok(path) = std::env::var("BF_LINEARIZABILITY_REPLAY") {
            shuttle::replay_from_file(run, path);
            return;
        }
        let schedules = std::path::Path::new("target").join("linearizability-schedules");
        std::fs::create_dir_all(&schedules).unwrap();
        for seed in [20261029, 400091] {
            let mut config = shuttle::Config::default();
            config.max_steps = shuttle::MaxSteps::FailAfter(200_000);
            config.failure_persistence = shuttle::FailurePersistence::File(Some(schedules.clone()));
            shuttle::Runner::new(
                shuttle::scheduler::RandomScheduler::new_from_seed(seed, 32),
                config,
            )
            .run(run);
        }
    }
}

#[cfg(not(feature = "shuttle"))]
#[test]
fn split_histories() {
    run_history(Scenario::Split);
}
