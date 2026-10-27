//! The manifest holds a bounded history and an open reads a bounded log.
//!
//! Every flush commits a new table version to the manifest. Kept for ever,
//! and with nothing to checkpoint the log behind them, the manifest grew with
//! every flush and every open read all of it: an ESM pulse archive flushed for
//! a week held 4,439 log segments and 6.2 GB, and spent 13.5 s of each start
//! reading them.
//!
//! These tests flush once per ingest, far faster than any real cadence, and
//! hold the manifest to fixed bounds however long they run.

use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use fusio::{disk::LocalFs, executor::tokio::TokioExecutor};
use tonbo::{
    db::{DB, DbBuilder},
    prelude::BatchesThreshold,
};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn batch(from: i64, n: i64) -> RecordBatch {
    let ids: Vec<String> = (from..from + n).map(|i| format!("key-{i:012}")).collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(Int64Array::from_iter_values(from..from + n)),
        ],
    )
    .unwrap()
}

/// Seal every batch and flush every sealed batch: one manifest version per
/// ingest, with no L0 compaction behind it.
async fn open(dir: &Path) -> DB<LocalFs, TokioExecutor> {
    DbBuilder::from_schema_key_name(schema(), "id")
        .unwrap()
        .on_disk(dir)
        .unwrap()
        .with_seal_policy(Arc::new(BatchesThreshold { batches: 1 }))
        .with_minor_compaction(1, 0)
        .open()
        .await
        .unwrap()
}

/// Files and bytes under the version manifest's log and checkpoints.
fn manifest_files(dir: &Path) -> (usize, u64) {
    fn walk(p: &Path, acc: &mut (usize, u64)) {
        let Ok(rd) = std::fs::read_dir(p) else {
            return;
        };
        for e in rd.flatten() {
            let path = e.path();
            if path.is_dir() {
                walk(&path, acc);
            } else if let Ok(m) = e.metadata() {
                acc.0 += 1;
                acc.1 += m.len();
            }
        }
    }
    let mut acc = (0, 0);
    walk(&dir.join("manifest").join("version"), &mut acc);
    acc
}

/// The version manifest's last commit: one per flush once the table is open.
fn commits(dir: &Path) -> u64 {
    std::fs::read_to_string(dir.join("manifest").join("version").join("head.json"))
        .ok()
        .and_then(|head| {
            let at = head.find("\"last_txn_id\":")? + "\"last_txn_id\":".len();
            let digits: String = head[at..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            digits.parse().ok()
        })
        .unwrap_or(0)
}

/// Ingest a batch and wait for its flush to commit, so every ingest is one
/// manifest version: minor compaction runs in the background and would
/// otherwise take several sealed batches in one flush.
async fn ingest_flushed(db: &DB<LocalFs, TokioExecutor>, dir: &Path, from: i64) {
    let before = commits(dir);
    db.ingest(batch(from, 50)).await.unwrap();
    while commits(dir) <= before {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

fn log_segments(dir: &Path) -> usize {
    std::fs::read_dir(dir.join("manifest").join("version").join("segments"))
        .map(|rd| rd.flatten().count())
        .unwrap_or(0)
}

async fn rows(db: &DB<LocalFs, TokioExecutor>) -> usize {
    db.scan()
        .collect()
        .await
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum()
}

/// The manifest holds a fixed number of versions however many flushes made
/// it. Each version names every SST, and without L0 compaction (which ESM's
/// table runs without) a table gains one per flush, so a version grows with
/// the table and a manifest of N versions with N times that: kept for ever,
/// N is the flush count and the manifest grows with its square (an ESM archive
/// reached 6.2 GB). Bounded, it grows only as one version does.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_long_run_of_flushes_keeps_a_bounded_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path()).await;
    let mut next = 0;
    // The largest the manifest gets over a stretch of 100 flushes, the
    // checkpoint's sawtooth included.
    let mut at_400 = (0, 0);
    let mut at_1200 = (0, 0);
    for i in 0..1200 {
        ingest_flushed(&db, dir.path(), next).await;
        next += 50;
        let now = manifest_files(dir.path());
        let stretch = match i {
            300..400 => &mut at_400,
            1100.. => &mut at_1200,
            _ => continue,
        };
        *stretch = (stretch.0.max(now.0), stretch.1.max(now.1));
    }
    assert_eq!(rows(&db).await, next as usize, "every row is readable");
    let versions = db.list_versions(0).await.unwrap();
    drop(db);

    eprintln!("manifest (files, bytes): {at_400:?} at 400 flushes, {at_1200:?} at 1,200");
    assert!(
        versions.len() <= 16,
        "the manifest keeps a window of versions, not all 1,200: {}",
        versions.len()
    );
    assert!(
        log_segments(dir.path()) <= 40,
        "the log behind the checkpoint is cut back: {} segments",
        log_segments(dir.path())
    );
    // Three times the SSTs, so three times the version: about three times the
    // manifest. Every version kept, it would be nine.
    assert!(
        at_1200.0 <= at_400.0 + 8,
        "the manifest's files do not accumulate: {at_400:?} (files, bytes) at 400 flushes, \
         {at_1200:?} at 1,200"
    );
    assert!(
        at_1200.1 <= at_400.1 * 9 / 2,
        "the manifest grows as one version does, not with every version: {at_400:?} (files, \
         bytes) at 400 flushes, {at_1200:?} at 1,200"
    );

    let db = open(dir.path()).await;
    assert_eq!(
        rows(&db).await,
        next as usize,
        "every row survives a reopen"
    );
}

/// Without L0 compaction every version names every SST, so a version grows
/// with the table; the log an open reads is still a checkpoint and a short
/// tail, and the open takes no longer after 1,200 flushes than after 200.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_open_reads_a_bounded_log_however_long_the_table_has_run() {
    let dir = tempfile::tempdir().unwrap();
    let mut next = 0;
    let mut opens: Vec<Duration> = Vec::new();
    for round in 0..6 {
        let db = open(dir.path()).await;
        for _ in 0..200 {
            ingest_flushed(&db, dir.path(), next).await;
            next += 50;
        }
        drop(db);
        let segments = log_segments(dir.path());
        assert!(
            segments <= 40,
            "round {round}: the log behind the checkpoint is cut back: {segments} segments"
        );
        let started = Instant::now();
        let db = open(dir.path()).await;
        opens.push(started.elapsed());
        assert_eq!(
            rows(&db).await,
            next as usize,
            "round {round}: every row survives"
        );
        let versions = db.list_versions(0).await.unwrap();
        assert!(
            versions.len() <= 16,
            "round {round}: {} versions",
            versions.len()
        );
        drop(db);
    }
    // An open reads the head, the checkpoint's runs and the tail: the same
    // amount of log after every round. What still grows is one version, with
    // the table's SST count, which L0 compaction would bound.
    eprintln!("opens after each 200 flushes: {opens:?}");
    let first = opens[0];
    let last = *opens.last().unwrap();
    assert!(
        last <= first * 4 + Duration::from_millis(250),
        "opens stay flat as the table ages: {opens:?}"
    );
}
