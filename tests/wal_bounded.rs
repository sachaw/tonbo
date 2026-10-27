//! The WAL holds only what is not yet in an SST.
//!
//! An append-only writer that ingests steadily, and lets minor compaction flush
//! behind it, must see its WAL stay bounded: every flushed segment is below the
//! floor and pruned. A floor that moves one segment per flush (or not at all)
//! falls behind as soon as more than one segment is written between two flushes,
//! the WAL grows without bound, and every open replays all of it into memory.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use fusio::{disk::LocalFs, executor::tokio::TokioExecutor};
use tonbo::db::{DB, DbBuilder};

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

fn wal_segments(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir.join("wal"))
        .map(|rd| {
            rd.filter(|e| {
                e.as_ref()
                    .is_ok_and(|e| e.path().extension().is_some_and(|x| x == "tonwal"))
            })
            .count()
        })
        .unwrap_or(0)
}

async fn open(dir: &std::path::Path) -> DB<LocalFs, TokioExecutor> {
    DbBuilder::from_schema_key_name(schema(), "id")
        .unwrap()
        .on_disk(dir)
        .unwrap()
        // Small segments, so one flush interval spans many of them.
        .wal_segment_bytes(16 << 10)
        .open()
        .await
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn steady_ingest_keeps_the_wal_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path()).await;
    // The fewest segments over the second half of the run: just after a flush,
    // the WAL is only what came after it.
    let mut trough = usize::MAX;
    let mut next = 0;
    for i in 0..1500 {
        db.ingest(batch(next, 200)).await.unwrap();
        next += 200;
        if i >= 750 {
            trough = trough.min(wal_segments(dir.path()));
        }
    }
    let end = wal_segments(dir.path());
    let total = db
        .scan()
        .collect()
        .await
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum::<usize>();
    assert_eq!(total, next as usize, "every row is readable");
    drop(db);

    // Everything is still there after an open, and the open replayed only the tail.
    let db = open(dir.path()).await;
    let total = db
        .scan()
        .collect()
        .await
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum::<usize>();
    assert_eq!(total, next as usize, "every row survives a reopen");
    // The default seal policy and minor-compaction threshold hold tens of
    // thousands of these rows unflushed, so the WAL is a sawtooth: it climbs to
    // a flush and drops back to what is still live. Unbounded, it only climbs,
    // to ~500 segments here.
    assert!(
        trough <= 60,
        "the WAL holds only what is not yet flushed: {trough} segments at fewest, {end} at the end"
    );
}
