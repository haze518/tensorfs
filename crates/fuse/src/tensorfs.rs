use fuser::experimental::{
    AsyncFilesystem, DirEntListBuilder, GetAttrResponse, LookupResponse, RequestContext,
    Result as FResult,
};
use fuser::{Errno, FileHandle, LockOwner, OpenFlags};
use fuser::{FileAttr, FileType, INodeNo};
use std::cmp::min;
use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::UNIX_EPOCH;
use tensorfs::cas::Cas;
use tensorfs::error::TensorFsError;
use tensorfs::manifest::Manifest;
use tensorfs::resolver::Resolver;
use tensorfs::source::RemoteSource;

use crate::error::to_errno;

const TTL: Duration = Duration::from_secs(1);

enum Node {
    Root,
    Model {
        id: String,
    },
    File {
        model_id: String,
        path: String,
        size: u64,
    },
}

struct INodesTable {
    next: AtomicU64,
    nodes: HashMap<u64, Node>,
    children: HashMap<u64, HashMap<OsString, u64>>,
    parents: HashMap<u64, u64>,
}

impl INodesTable {
    const ROOT_INO: u64 = 1;

    fn new() -> Self {
        let mut nodes = HashMap::new();
        nodes.insert(Self::ROOT_INO, Node::Root);

        let mut parents = HashMap::new();
        parents.insert(Self::ROOT_INO, Self::ROOT_INO);

        Self {
            next: AtomicU64::new(2),
            nodes,
            children: HashMap::new(),
            parents,
        }
    }

    fn add_child(&mut self, parent: u64, name: OsString, node: Node) -> u64 {
        let ino = self.next.fetch_add(1, Ordering::Relaxed);

        self.nodes.insert(ino, node);
        self.children.entry(parent).or_default().insert(name, ino);
        self.parents.insert(ino, parent);

        ino
    }

    fn lookup(&self, parent: u64, name: &OsStr) -> Option<(u64, &Node)> {
        let child_ino = self.children.get(&parent)?.get(name)?;
        let node = self.nodes.get(child_ino)?;

        Some((*child_ino, node))
    }

    fn get(&self, ino: u64) -> Option<&Node> {
        self.nodes.get(&ino)
    }

    fn parent(&self, ino: u64) -> Option<u64> {
        self.parents.get(&ino).copied()
    }

    fn children(&self, ino: u64) -> impl Iterator<Item = (&OsString, &u64)> {
        self.children
            .get(&ino)
            .into_iter()
            .flat_map(|children| children.iter())
    }
}

pub struct TensorFs<C, R> {
    path: PathBuf,
    cas: C,
    remote: R,
    table: INodesTable,
    manifests: HashMap<String, Manifest>,
    resolver: Resolver,
}

impl<C: Cas, R: RemoteSource> TensorFs<C, R> {
    pub fn new(path: &Path, cas: C, remote: R) -> Result<Self, TensorFsError> {
        if !path.is_dir() {
            return Err(TensorFsError::BadRequest);
        }

        let mut manifests = HashMap::new();
        let mut table = INodesTable::new();

        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let model_name = entry.file_name();
            let model_id = model_name.to_string_lossy().to_string();

            let manifest_path = path.join(&model_name);
            let manifest = Manifest::load(&manifest_path)?;

            let model_ino = table.add_child(
                INodesTable::ROOT_INO,
                model_name,
                Node::Model {
                    id: model_id.clone(),
                },
            );

            for file in &manifest.files {
                table.add_child(
                    model_ino,
                    OsString::from(file.path.clone()),
                    Node::File {
                        model_id: model_id.clone(),
                        path: file.path.clone(),
                        size: file.size,
                    },
                );
            }

            manifests.insert(model_id, manifest);
        }

        Ok(TensorFs {
            path: path.to_path_buf(),
            cas,
            remote,
            table,
            manifests,
            resolver: Resolver {},
        })
    }
}

#[async_trait::async_trait]
impl<C: Cas, R: RemoteSource> AsyncFilesystem for TensorFs<C, R> {
    async fn lookup(
        &self,
        _context: &RequestContext,
        parent: INodeNo,
        name: &OsStr,
    ) -> FResult<LookupResponse> {
        let (ino, node) = self
            .table
            .lookup(parent.into(), name)
            .ok_or(Errno::ENOENT)?;

        let attr = attr_for_node(ino, node);

        Ok(LookupResponse::new(TTL, attr, fuser::Generation(0)))
    }

    async fn getattr(
        &self,
        _context: &RequestContext,
        ino: INodeNo,
        _file_handle: Option<fuser::FileHandle>,
    ) -> FResult<GetAttrResponse> {
        let node = match self.table.get(ino.into()) {
            Some(node) => node,
            None => return Err(Errno::ENOENT),
        };
        let attr = attr_for_node(ino.into(), node);
        Ok(GetAttrResponse::new(TTL, attr))
    }

    async fn read(
        &self,
        _context: &RequestContext,
        ino: INodeNo,
        _file_handle: FileHandle,
        offset: u64,
        size: u32,
        _flags: OpenFlags,
        _lock: Option<LockOwner>,
        out_data: &mut Vec<u8>,
    ) -> FResult<()> {
        let node = self.table.get(ino.into()).ok_or(Errno::ENOENT)?;

        let (model_id, path, _) = if let Node::File {
            model_id,
            path,
            size,
        } = node
        {
            (model_id, path, size)
        } else {
            return Err(Errno::EISDIR);
        };

        let manifest = self.manifests.get(model_id).ok_or(Errno::EIO)?;

        let file = manifest
            .files
            .iter()
            .find_map(|x| {
                if &x.path == path {
                    return Some(x);
                }
                None
            })
            .ok_or(Errno::ENOENT)?;

        if offset >= file.size {
            return Ok(());
        }

        let len = min(size as u64, file.size - offset);
        let data = self
            .resolver
            .resolve(file, offset, len)
            .map_err(|e| to_errno(e))?;

        for d in data {
            let val = self
                .cas
                .read_range(d.chunk_id, d.chunk_offset, d.len as usize)
                .await
                .map_err(|e| to_errno(e))?;
            out_data.extend_from_slice(&val);
        }

        Ok(())
    }

    async fn readdir(
        &self,
        _context: &RequestContext,
        ino: INodeNo,
        _file_handle: FileHandle,
        offset: u64,
        mut builder: DirEntListBuilder<'_>,
    ) -> FResult<()> {
        let node = self.table.get(ino.into()).ok_or(Errno::ENOENT)?;

        match node {
            Node::Root | Node::Model { .. } => {
                if offset == 0 {
                    if builder.add(ino, 1, FileType::Directory, ".") {
                        return Ok(());
                    }
                }

                if offset <= 1 {
                    let parent_ino = self.table.parent(ino.into()).ok_or(Errno::ENOENT)?;

                    if builder.add(INodeNo(parent_ino), 2, FileType::Directory, "..") {
                        return Ok(());
                    }
                }

                for (i, (name, child_ino)) in self
                    .table
                    .children(ino.into())
                    .skip(offset.saturating_sub(2) as usize)
                    .enumerate()
                {
                    let child_node = self.table.get(*child_ino).ok_or(Errno::ENOENT)?;
                    let attr = attr_for_node(*child_ino, child_node);

                    let next_offset = 3 + i as u64;
                    if builder.add(INodeNo(*child_ino), next_offset, attr.kind, name) {
                        break;
                    }
                }
            }
            Node::File { .. } => return Err(Errno::ENOTDIR),
        }

        Ok(())
    }
}

fn attr_for_node(ino: u64, node: &Node) -> FileAttr {
    let (size, kind, perm, nlink) = match node {
        Node::Root => (0, FileType::Directory, 0o555, 2),
        Node::Model { .. } => (0, FileType::Directory, 0o555, 2),
        Node::File { size, .. } => (*size, FileType::RegularFile, 0o444, 1),
    };

    FileAttr {
        ino: INodeNo(ino),
        size,
        kind,
        perm,
        blocks: size.div_ceil(512),
        uid: unsafe { libc::getuid() },
        gid: unsafe { libc::getgid() },
        atime: UNIX_EPOCH,
        mtime: UNIX_EPOCH,
        ctime: UNIX_EPOCH,
        crtime: UNIX_EPOCH,
        nlink,
        blksize: 4096,
        rdev: 0,
        flags: 0,
    }
}
