use crate::chunk::ChunkId;
use crate::error::TensorFsError;
use crate::manifest::File;

#[derive(Debug)]
pub struct ResolvedSlice {
    chunk_id: ChunkId,
    chunk_offset: u64,
    len: u64,
}

pub struct Resolver {}

impl Resolver {
    fn resolve_segment(
        &self,
        file: &File,
        offset: u64,
        len: u64,
    ) -> Result<ResolvedSlice, TensorFsError> {
        for segment in &file.segments {
            if segment.file_offset <= offset && offset < segment.file_offset + segment.len {
                let delta = offset - segment.file_offset;
                let chunk_offset = segment.chunk_offset + delta;
                let available = segment.len - delta;
                let resolved_slice_len = len.min(available);

                return Ok(ResolvedSlice {
                    chunk_id: segment.chunk_id,
                    chunk_offset,
                    len: resolved_slice_len,
                });
            }
        }

        Err(TensorFsError::SegmentNotFound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{File, Segment};

    fn chunk_id(byte: u8) -> ChunkId {
        ChunkId::from_bytes([byte; 32])
    }

    fn file_with_segments(segments: Vec<Segment>) -> File {
        File {
            path: "test.bin".to_string(),
            size: 10_000,
            segments,
        }
    }

    #[test]
    fn resolve_segment_returns_slice_inside_single_segment() {
        let resolver = Resolver {};
        let cid = chunk_id(1);

        let file = file_with_segments(vec![
            Segment {
                chunk_id: cid,
                file_offset: 100,
                chunk_offset: 20,
                len: 50,
            },
        ]);

        let resolved = resolver.resolve_segment(&file, 110, 10).unwrap();

        assert_eq!(resolved.chunk_id, cid);
        assert_eq!(resolved.chunk_offset, 30); // 20 + (110 - 100)
        assert_eq!(resolved.len, 10);
    }

    #[test]
    fn resolve_segment_truncates_len_at_segment_end() {
        let resolver = Resolver {};
        let cid = chunk_id(2);

        let file = file_with_segments(vec![
            Segment {
                chunk_id: cid,
                file_offset: 100,
                chunk_offset: 20,
                len: 50,
            },
        ]);

        let resolved = resolver.resolve_segment(&file, 140, 20).unwrap();

        assert_eq!(resolved.chunk_id, cid);
        assert_eq!(resolved.chunk_offset, 60); // 20 + (140 - 100)
        assert_eq!(resolved.len, 10);
    }

    #[test]
    fn resolve_segment_accepts_offset_at_segment_start() {
        let resolver = Resolver {};
        let cid = chunk_id(3);

        let file = file_with_segments(vec![
            Segment {
                chunk_id: cid,
                file_offset: 500,
                chunk_offset: 1000,
                len: 25,
            },
        ]);

        let resolved = resolver.resolve_segment(&file, 500, 7).unwrap();

        assert_eq!(resolved.chunk_id, cid);
        assert_eq!(resolved.chunk_offset, 1000);
        assert_eq!(resolved.len, 7);
    }

    #[test]
    fn resolve_segment_rejects_offset_at_segment_end() {
        let resolver = Resolver {};

        let file = file_with_segments(vec![
            Segment {
                chunk_id: chunk_id(4),
                file_offset: 500,
                chunk_offset: 1000,
                len: 25,
            },
        ]);

        let err = resolver.resolve_segment(&file, 525, 1).unwrap_err();

        assert!(matches!(err, TensorFsError::SegmentNotFound));
    }

    #[test]
    fn resolve_segment_returns_not_found_when_offset_before_all_segments() {
        let resolver = Resolver {};

        let file = file_with_segments(vec![
            Segment {
                chunk_id: chunk_id(5),
                file_offset: 100,
                chunk_offset: 0,
                len: 10,
            },
        ]);

        let err = resolver.resolve_segment(&file, 99, 1).unwrap_err();

        assert!(matches!(err, TensorFsError::SegmentNotFound));
    }

    #[test]
    fn resolve_segment_returns_not_found_when_offset_after_all_segments() {
        let resolver = Resolver {};

        let file = file_with_segments(vec![
            Segment {
                chunk_id: chunk_id(6),
                file_offset: 100,
                chunk_offset: 0,
                len: 10,
            },
        ]);

        let err = resolver.resolve_segment(&file, 1000, 1).unwrap_err();

        assert!(matches!(err, TensorFsError::SegmentNotFound));
    }

    #[test]
    fn resolve_segment_uses_correct_segment_when_file_has_multiple_segments() {
        let resolver = Resolver {};
        let cid1 = chunk_id(7);
        let cid2 = chunk_id(8);

        let file = file_with_segments(vec![
            Segment {
                chunk_id: cid1,
                file_offset: 0,
                chunk_offset: 0,
                len: 100,
            },
            Segment {
                chunk_id: cid2,
                file_offset: 100,
                chunk_offset: 500,
                len: 50,
            },
        ]);

        let resolved = resolver.resolve_segment(&file, 120, 10).unwrap();

        assert_eq!(resolved.chunk_id, cid2);
        assert_eq!(resolved.chunk_offset, 520); // 500 + (120 - 100)
        assert_eq!(resolved.len, 10);
    }
}