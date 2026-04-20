use serde::de::Error as DeError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct ChunkId([u8; 32]);

impl ChunkId {
    pub const ZERO: Self = Self([0; 32]);

    pub fn to_hex(&self) -> String {
        let mut s = String::new();
        let table = b"0123456789abcdef";
        for &b in self.0.iter() {
            s.push(table[(b >> 4) as usize] as char);
            s.push(table[(b & 0xf) as usize] as char);
        }
        s
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        *self == Self::ZERO
    }
}

impl Default for ChunkId {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.as_bytes())
    }
}

impl Serialize for ChunkId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for ChunkId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;

        let bytes = hex::decode(&s).map_err(|_| D::Error::custom("invalid hex"))?;

        if bytes.len() != 32 {
            return Err(D::Error::custom("invalid length for ChunkId"));
        }

        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);

        Ok(ChunkId::from_bytes(arr))
    }
}
