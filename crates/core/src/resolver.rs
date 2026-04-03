use crate::chunk::ChunkId;
use crate::error::TensorFsError;
use crate::manifest::{File, Segment};

#[derive(Debug)]
pub struct ResolvedSlice {
    chunk_id: ChunkId,
    chunk_offset: u64,
    len: u64,
}

pub struct Resolver {}

impl Resolver {
    fn resolve(
        &self,
        file: &File,
        offset: u64,
        len: u64,
    ) -> Result<Vec<ResolvedSlice>, TensorFsError> {
        if len == 0 {
            return Ok(Vec::new());
        }

        let end = offset
            .checked_add(len)
            .ok_or(TensorFsError::IncorrectReadInterval)?;

        if end > file.size {
            return Err(TensorFsError::IncorrectReadInterval);
        }

        let mut ix = self.find_first(file, offset)?;
        let mut current_offset = offset;
        let mut remaining = len;
        let mut result = Vec::new();

        while let Some(segment) = file.segments.get(ix) {
            let resolved = match self.resolve_segment(segment, current_offset, remaining) {
                Ok(resolved) => resolved,
                Err(_) => break,
            };

            current_offset += resolved.len;
            remaining -= resolved.len;
            result.push(resolved);

            if remaining == 0 {
                return Ok(result);
            }

            ix += 1;
        }

        Err(TensorFsError::IncorrectReadInterval)
    }

    fn find_first(&self, file: &File, offset: u64) -> Result<usize, TensorFsError> {
        file.segments
            .iter()
            .position(|segment| Self::contains_offset(segment, offset))
            .ok_or(TensorFsError::SegmentNotFound)
    }

    fn resolve_segment(
        &self,
        segment: &Segment,
        offset: u64,
        len: u64,
    ) -> Result<ResolvedSlice, TensorFsError> {
        if !Self::contains_offset(segment, offset) {
            return Err(TensorFsError::SegmentNotFound);
        }

        let delta = offset - segment.file_offset;
        let chunk_offset = segment.chunk_offset + delta;
        let available = segment.len - delta;
        let resolved_slice_len = len.min(available);

        Ok(ResolvedSlice {
            chunk_id: segment.chunk_id,
            chunk_offset,
            len: resolved_slice_len,
        })
    }

    fn contains_offset(segment: &Segment, offset: u64) -> bool {
        segment.file_offset <= offset && offset < segment.file_offset + segment.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{File, Segment};

    fn chunk_id(byte: u8) -> ChunkId {
        ChunkId::from_bytes([byte; 32])
    }

    fn file_with_segments(size: u64, segments: Vec<Segment>) -> File {
        File {
            path: "test.bin".to_string(),
            size,
            segments,
        }
    }

    #[test]
    fn resolve_returns_single_slice_when_range_fits_one_segment() {
        let resolver = Resolver {};
        let cid = chunk_id(1);

        let file = file_with_segments(
            100,
            vec![Segment {
                chunk_id: cid,
                file_offset: 0,
                chunk_offset: 10,
                len: 100,
            }],
        );

        let slices = resolver.resolve(&file, 20, 15).unwrap();

        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].chunk_id, cid);
        assert_eq!(slices[0].chunk_offset, 30);
        assert_eq!(slices[0].len, 15);
    }

    #[test]
    fn resolve_splits_read_between_two_segments() {
        let resolver = Resolver {};
        let cid1 = chunk_id(1);
        let cid2 = chunk_id(2);

        let file = file_with_segments(
            200,
            vec![
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
                    len: 100,
                },
            ],
        );

        let slices = resolver.resolve(&file, 80, 40).unwrap();

        assert_eq!(slices.len(), 2);

        assert_eq!(slices[0].chunk_id, cid1);
        assert_eq!(slices[0].chunk_offset, 80);
        assert_eq!(slices[0].len, 20);

        assert_eq!(slices[1].chunk_id, cid2);
        assert_eq!(slices[1].chunk_offset, 500);
        assert_eq!(slices[1].len, 20);
    }

    #[test]
    fn resolve_splits_read_between_three_segments() {
        let resolver = Resolver {};
        let cid1 = chunk_id(1);
        let cid2 = chunk_id(2);
        let cid3 = chunk_id(3);

        let file = file_with_segments(
            300,
            vec![
                Segment {
                    chunk_id: cid1,
                    file_offset: 0,
                    chunk_offset: 0,
                    len: 100,
                },
                Segment {
                    chunk_id: cid2,
                    file_offset: 100,
                    chunk_offset: 1000,
                    len: 100,
                },
                Segment {
                    chunk_id: cid3,
                    file_offset: 200,
                    chunk_offset: 2000,
                    len: 100,
                },
            ],
        );

        let slices = resolver.resolve(&file, 50, 180).unwrap();

        assert_eq!(slices.len(), 3);

        assert_eq!(slices[0].chunk_id, cid1);
        assert_eq!(slices[0].chunk_offset, 50);
        assert_eq!(slices[0].len, 50);

        assert_eq!(slices[1].chunk_id, cid2);
        assert_eq!(slices[1].chunk_offset, 1000);
        assert_eq!(slices[1].len, 100);

        assert_eq!(slices[2].chunk_id, cid3);
        assert_eq!(slices[2].chunk_offset, 2000);
        assert_eq!(slices[2].len, 30);
    }

    #[test]
    fn resolve_starts_exactly_at_next_segment_boundary() {
        let resolver = Resolver {};
        let cid1 = chunk_id(1);
        let cid2 = chunk_id(2);

        let file = file_with_segments(
            200,
            vec![
                Segment {
                    chunk_id: cid1,
                    file_offset: 0,
                    chunk_offset: 10,
                    len: 100,
                },
                Segment {
                    chunk_id: cid2,
                    file_offset: 100,
                    chunk_offset: 20,
                    len: 100,
                },
            ],
        );

        let slices = resolver.resolve(&file, 100, 30).unwrap();

        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].chunk_id, cid2);
        assert_eq!(slices[0].chunk_offset, 20);
        assert_eq!(slices[0].len, 30);
    }

    #[test]
    fn resolve_reads_until_end_of_last_segment() {
        let resolver = Resolver {};
        let cid1 = chunk_id(1);
        let cid2 = chunk_id(2);

        let file = file_with_segments(
            150,
            vec![
                Segment {
                    chunk_id: cid1,
                    file_offset: 0,
                    chunk_offset: 0,
                    len: 100,
                },
                Segment {
                    chunk_id: cid2,
                    file_offset: 100,
                    chunk_offset: 1000,
                    len: 50,
                },
            ],
        );

        let slices = resolver.resolve(&file, 90, 60).unwrap();

        assert_eq!(slices.len(), 2);

        assert_eq!(slices[0].chunk_id, cid1);
        assert_eq!(slices[0].chunk_offset, 90);
        assert_eq!(slices[0].len, 10);

        assert_eq!(slices[1].chunk_id, cid2);
        assert_eq!(slices[1].chunk_offset, 1000);
        assert_eq!(slices[1].len, 50);
    }

    #[test]
    fn resolve_returns_segment_not_found_when_offset_is_before_first_segment() {
        let resolver = Resolver {};

        let file = file_with_segments(
            100,
            vec![Segment {
                chunk_id: chunk_id(1),
                file_offset: 10,
                chunk_offset: 0,
                len: 50,
            }],
        );

        let err = resolver.resolve(&file, 0, 10).unwrap_err();

        assert!(matches!(err, TensorFsError::SegmentNotFound));
    }

    #[test]
    fn resolve_returns_segment_not_found_when_offset_is_after_all_segments_but_still_within_file_size()
     {
        let resolver = Resolver {};

        let file = file_with_segments(
            100,
            vec![Segment {
                chunk_id: chunk_id(1),
                file_offset: 0,
                chunk_offset: 0,
                len: 50,
            }],
        );

        let err = resolver.resolve(&file, 60, 10).unwrap_err();

        assert!(matches!(err, TensorFsError::SegmentNotFound));
    }

    #[test]
    fn resolve_returns_incorrect_read_interval_when_there_is_a_gap_between_segments() {
        let resolver = Resolver {};
        let cid1 = chunk_id(1);

        let file = file_with_segments(
            300,
            vec![
                Segment {
                    chunk_id: cid1,
                    file_offset: 0,
                    chunk_offset: 0,
                    len: 100,
                },
                Segment {
                    chunk_id: chunk_id(2),
                    file_offset: 200,
                    chunk_offset: 1000,
                    len: 100,
                },
            ],
        );

        let err = resolver.resolve(&file, 50, 200).unwrap_err();

        assert!(matches!(err, TensorFsError::IncorrectReadInterval));
    }

    #[test]
    fn resolve_returns_empty_vec_for_zero_len_read() {
        let resolver = Resolver {};

        let file = file_with_segments(
            100,
            vec![Segment {
                chunk_id: chunk_id(1),
                file_offset: 0,
                chunk_offset: 0,
                len: 100,
            }],
        );

        let slices = resolver.resolve(&file, 10, 0).unwrap();

        assert!(slices.is_empty());
    }

    #[test]
    fn resolve_produces_contiguous_slices_without_gaps_or_overlaps() {
        let resolver = Resolver {};
        let cid1 = chunk_id(1);
        let cid2 = chunk_id(2);
        let cid3 = chunk_id(3);

        let file = file_with_segments(
            300,
            vec![
                Segment {
                    chunk_id: cid1,
                    file_offset: 0,
                    chunk_offset: 0,
                    len: 100,
                },
                Segment {
                    chunk_id: cid2,
                    file_offset: 100,
                    chunk_offset: 1000,
                    len: 100,
                },
                Segment {
                    chunk_id: cid3,
                    file_offset: 200,
                    chunk_offset: 2000,
                    len: 100,
                },
            ],
        );

        let offset = 40;
        let len = 220;

        let slices = resolver.resolve(&file, offset, len).unwrap();

        assert_eq!(slices.len(), 3);

        let total_len: u64 = slices.iter().map(|s| s.len).sum();
        assert_eq!(total_len, len);

        assert_eq!(slices[0].len, 60);
        assert_eq!(slices[1].len, 100);
        assert_eq!(slices[2].len, 60);
    }

    #[test]
    fn resolve_returns_incorrect_read_interval_when_range_exceeds_file_size() {
        let resolver = Resolver {};

        let file = file_with_segments(
            100,
            vec![Segment {
                chunk_id: chunk_id(1),
                file_offset: 0,
                chunk_offset: 0,
                len: 100,
            }],
        );

        let err = resolver.resolve(&file, 90, 20).unwrap_err();

        assert!(matches!(err, TensorFsError::IncorrectReadInterval));
    }

    #[test]
    fn resolve_returns_incorrect_read_interval_when_offset_plus_len_overflows() {
        let resolver = Resolver {};

        let file = file_with_segments(
            u64::MAX,
            vec![Segment {
                chunk_id: chunk_id(1),
                file_offset: 0,
                chunk_offset: 0,
                len: u64::MAX,
            }],
        );

        let err = resolver.resolve(&file, u64::MAX - 5, 10).unwrap_err();

        assert!(matches!(err, TensorFsError::IncorrectReadInterval));
    }

    #[test]
    fn resolve_segment_returns_slice_inside_single_segment() {
        let resolver = Resolver {};
        let cid = chunk_id(1);

        let segment = Segment {
            chunk_id: cid,
            file_offset: 100,
            chunk_offset: 20,
            len: 50,
        };

        let resolved = resolver.resolve_segment(&segment, 110, 10).unwrap();

        assert_eq!(resolved.chunk_id, cid);
        assert_eq!(resolved.chunk_offset, 30);
        assert_eq!(resolved.len, 10);
    }

    #[test]
    fn resolve_segment_truncates_len_at_segment_end() {
        let resolver = Resolver {};
        let cid = chunk_id(2);

        let segment = Segment {
            chunk_id: cid,
            file_offset: 100,
            chunk_offset: 20,
            len: 50,
        };

        let resolved = resolver.resolve_segment(&segment, 140, 20).unwrap();

        assert_eq!(resolved.chunk_id, cid);
        assert_eq!(resolved.chunk_offset, 60);
        assert_eq!(resolved.len, 10);
    }

    #[test]
    fn resolve_segment_accepts_offset_at_segment_start() {
        let resolver = Resolver {};
        let cid = chunk_id(3);

        let segment = Segment {
            chunk_id: cid,
            file_offset: 500,
            chunk_offset: 1000,
            len: 25,
        };

        let resolved = resolver.resolve_segment(&segment, 500, 7).unwrap();

        assert_eq!(resolved.chunk_id, cid);
        assert_eq!(resolved.chunk_offset, 1000);
        assert_eq!(resolved.len, 7);
    }

    #[test]
    fn resolve_segment_rejects_offset_at_segment_end() {
        let resolver = Resolver {};

        let segment = Segment {
            chunk_id: chunk_id(4),
            file_offset: 500,
            chunk_offset: 1000,
            len: 25,
        };

        let err = resolver.resolve_segment(&segment, 525, 1).unwrap_err();

        assert!(matches!(err, TensorFsError::SegmentNotFound));
    }

    #[test]
    fn find_first_returns_index_of_matching_segment() {
        let resolver = Resolver {};

        let file = file_with_segments(
            300,
            vec![
                Segment {
                    chunk_id: chunk_id(1),
                    file_offset: 0,
                    chunk_offset: 0,
                    len: 100,
                },
                Segment {
                    chunk_id: chunk_id(2),
                    file_offset: 100,
                    chunk_offset: 1000,
                    len: 100,
                },
                Segment {
                    chunk_id: chunk_id(3),
                    file_offset: 200,
                    chunk_offset: 2000,
                    len: 100,
                },
            ],
        );

        let ix = resolver.find_first(&file, 150).unwrap();

        assert_eq!(ix, 1);
    }

    #[test]
    fn find_first_returns_not_found_when_no_segment_contains_offset() {
        let resolver = Resolver {};

        let file = file_with_segments(
            300,
            vec![
                Segment {
                    chunk_id: chunk_id(1),
                    file_offset: 0,
                    chunk_offset: 0,
                    len: 100,
                },
                Segment {
                    chunk_id: chunk_id(2),
                    file_offset: 200,
                    chunk_offset: 1000,
                    len: 100,
                },
            ],
        );

        let err = resolver.find_first(&file, 150).unwrap_err();

        assert!(matches!(err, TensorFsError::SegmentNotFound));
    }
}
