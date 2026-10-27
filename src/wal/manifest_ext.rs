//! Helpers that bridge WAL metadata into manifest structures.

use std::sync::Arc;

use crate::{
    manifest::WalSegmentRef,
    wal::{
        WalConfig, WalError,
        storage::{SegmentFrameBounds, WalStorage},
        wal_segment_file_id,
    },
};

/// The WAL segments on disk, oldest first, each with the frames it holds.
///
/// Read BEFORE the live frames are ([`segment_refs`]): a frame written after this
/// listing is newer than every frame in it, so it lands past whatever floor the
/// listing yields and is kept.
pub(crate) async fn wal_segments(
    cfg: &WalConfig,
) -> Result<Vec<(u64, SegmentFrameBounds)>, WalError> {
    let storage = WalStorage::new(Arc::clone(&cfg.segment_backend), cfg.dir.clone());
    let wal_state_hint = storage
        .load_state_handle(cfg.state_store.as_ref())
        .await?
        .and_then(|handle| handle.state().last_segment_seq);

    let Some(tail) = storage.tail_metadata_with_hint(wal_state_hint).await? else {
        return Ok(Vec::new());
    };

    let mut segments = Vec::with_capacity(tail.completed.len() + 1);
    for descriptor in tail
        .completed
        .iter()
        .chain(std::iter::once(&tail.active))
        .filter(|descriptor| descriptor.bytes > 0)
    {
        let bounds = storage
            .segment_frame_bounds(&descriptor.path)
            .await?
            .ok_or(WalError::Corrupt(
                "wal segment contained no frames despite non-zero length",
            ))?;
        segments.push((descriptor.seq, bounds));
    }
    segments.sort_by_key(|(seq, _)| *seq);
    segments.dedup_by_key(|(seq, _)| *seq);
    Ok(segments)
}

/// The segments a version keeps: those holding a frame not yet in an SST.
///
/// `live_frame_floor` is the oldest such frame. The first segment kept starts at
/// it, so a replay skips that segment's frames already flushed, and every segment
/// before it is below the floor and pruned. With nothing live the floor is the
/// frame after the last one written: the newest segment stays (a writer may still
/// be appending to it) and a replay starts past everything in it.
///
/// The floor is where the oldest live frame is, whatever it was before: it does
/// not creep forward a segment per flush, which fell behind any writer filling
/// more than one segment between two flushes and left the WAL, and every replay
/// of it, growing without bound.
pub(crate) fn segment_refs(
    segments: &[(u64, SegmentFrameBounds)],
    live_frame_floor: Option<u64>,
) -> Vec<WalSegmentRef> {
    let Some(&(newest, newest_bounds)) = segments.last() else {
        return Vec::new();
    };
    let live = live_frame_floor.filter(|&frame| frame <= newest_bounds.last_seq);
    let Some(live) = live else {
        return vec![WalSegmentRef::new(
            newest,
            wal_segment_file_id(newest),
            newest_bounds.last_seq.saturating_add(1),
            newest_bounds.last_seq,
        )];
    };
    let mut refs: Vec<WalSegmentRef> = segments
        .iter()
        .filter(|(_, bounds)| bounds.last_seq >= live)
        .map(|&(seq, bounds)| wal_segment_ref(seq, bounds))
        .collect();
    if let Some(first) = refs.first_mut()
        && live > first.first_frame()
    {
        *first = WalSegmentRef::new(first.seq(), *first.file_id(), live, first.last_frame());
    }
    refs
}

/// The segments to keep given the frames live now; see [`segment_refs`].
pub(crate) async fn collect_wal_segment_refs(
    cfg: &WalConfig,
    live_frame_floor: impl FnOnce() -> Option<u64>,
) -> Result<Vec<WalSegmentRef>, WalError> {
    let segments = wal_segments(cfg).await?;
    Ok(segment_refs(&segments, live_frame_floor()))
}

fn wal_segment_ref(seq: u64, bounds: SegmentFrameBounds) -> WalSegmentRef {
    WalSegmentRef::new(
        seq,
        wal_segment_file_id(seq),
        bounds.first_seq,
        bounds.last_seq,
    )
}

/// Prune WAL segments whose sequence is strictly below the manifest floor.
pub(crate) async fn prune_wal_segments(
    cfg: &WalConfig,
    floor: &WalSegmentRef,
) -> Result<usize, WalError> {
    let storage = WalStorage::new(Arc::clone(&cfg.segment_backend), cfg.dir.clone());
    if cfg.prune_dry_run {
        let segments = storage.list_segments_with_hint(None).await?;
        let removable = segments
            .into_iter()
            .filter(|descriptor| descriptor.seq < floor.seq())
            .count();
        Ok(removable)
    } else {
        storage.prune_below(floor.seq()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(seq: u64, first_seq: u64, last_seq: u64) -> (u64, SegmentFrameBounds) {
        (
            seq,
            SegmentFrameBounds {
                first_seq,
                last_seq,
            },
        )
    }

    fn floor(refs: &[WalSegmentRef]) -> (u64, u64) {
        (refs[0].seq(), refs[0].first_frame())
    }

    #[test]
    fn the_floor_is_the_oldest_live_frame_however_far_it_moved() {
        let segments = [seg(1, 0, 9), seg(2, 10, 19), seg(3, 20, 29), seg(4, 30, 39)];
        let refs = segment_refs(&segments, Some(25));
        assert_eq!(floor(&refs), (3, 25), "three segments on, in one flush");
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn nothing_live_keeps_only_the_newest_segment_and_replays_none_of_it() {
        let segments = [seg(1, 0, 9), seg(2, 10, 19)];
        let refs = segment_refs(&segments, None);
        assert_eq!(refs.len(), 1);
        assert_eq!(floor(&refs), (2, 20));
    }

    #[test]
    fn a_live_frame_newer_than_the_listing_is_past_its_floor() {
        let segments = [seg(1, 0, 9), seg(2, 10, 19)];
        assert_eq!(floor(&segment_refs(&segments, Some(20))), (2, 20));
    }

    #[test]
    fn a_live_frame_at_a_segment_start_keeps_it_whole() {
        let segments = [seg(1, 0, 9), seg(2, 10, 19)];
        assert_eq!(floor(&segment_refs(&segments, Some(10))), (2, 10));
        assert_eq!(segment_refs(&segments, Some(3)).len(), 2);
    }

    #[test]
    fn no_wal_no_segments() {
        assert!(segment_refs(&[], Some(1)).is_empty());
    }
}
