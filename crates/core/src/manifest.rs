use crate::chunk::ChunkId;

/// Example JSON:
/// {
///   "version": 1,
///   "source": "...",
///   "files": [
///     {
///       "path": "...",
///       "size": 123,
///       "segments": [
///         {
///           "chunk_id": "...",
///           "file_offset": 0,
///           "chunk_offset": 0,
///           "len": 123
///         }
///       ]
///     }
///   ]
/// }
pub struct Manifest {
    version: u16,
    source: String,
    files: Vec<File>,
}

pub struct File {
    path: String,
    size: u64,
    segments: Vec<Segment>,
}

pub struct Segment {
    chunk_id: ChunkId,
    file_offset: u64,
    chunk_offset: u64,
    len: u64,
}
