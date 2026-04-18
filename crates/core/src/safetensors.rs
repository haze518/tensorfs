use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::TensorFsError;

const SAFETENSORS_HEADER_LEN: usize = 8;

#[derive(Debug, Serialize, Deserialize)]
pub struct Safetensor {
    meta: Vec<TensorMeta>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorMeta {
    name: String,
    dtype: Dtype,
    shape: Vec<u64>,
    offset: u64,
    length: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I64,
    I32,
    I8,
    U8,
}

impl Dtype {
    pub fn size_in_bytes(self) -> u64 {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
            Dtype::BF16 => 2,
            Dtype::I64 => 8,
            Dtype::I32 => 4,
            Dtype::I8 => 1,
            Dtype::U8 => 1,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawTensorInfo {
    dtype: Dtype,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum HeaderEntry {
    Metadata(BTreeMap<String, String>),
    Tensor(RawTensorInfo),
}

pub fn parse_header(bytes: &[u8]) -> Result<Vec<TensorMeta>, TensorFsError> {
    if bytes.len() < SAFETENSORS_HEADER_LEN {
        return Err(TensorFsError::TooShortForHeaderLen);
    }

    let header_len_bytes: [u8; SAFETENSORS_HEADER_LEN] = bytes[0..SAFETENSORS_HEADER_LEN]
        .try_into()
        .map_err(|_| TensorFsError::IncorrectSafetensorsLen)?;
    let header_len_u64 = u64::from_le_bytes(header_len_bytes);
    let header_len =
        usize::try_from(header_len_u64).map_err(|_| TensorFsError::HeaderLenOverflow)?;

    let json_start = SAFETENSORS_HEADER_LEN;
    let json_end = json_start
        .checked_add(header_len)
        .ok_or(TensorFsError::HeaderLenOverflow)?;

    if bytes.len() < json_end {
        return Err(TensorFsError::IncompleteHeader);
    }

    let json_data = &bytes[json_start..json_end];

    let header: BTreeMap<String, HeaderEntry> =
        serde_json::from_slice(json_data).map_err(|_| TensorFsError::InvalidHeaderJson)?;

    let data_section_start = u64::try_from(json_end).map_err(|_| TensorFsError::OffsetOverflow)?;

    let mut result = Vec::new();

    for (name, entry) in header {
        if name == "__metadata__" {
            continue;
        }

        let raw = match entry {
            HeaderEntry::Tensor(t) => t,
            HeaderEntry::Metadata(_) => continue,
        };

        let [start, end] = raw.data_offsets;
        if end < start {
            return Err(TensorFsError::InvalidOffsets);
        }

        let length = end - start;

        let elements = raw
            .shape
            .iter()
            .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
            .ok_or(TensorFsError::InvalidTensorSize)?;

        let expected_len = elements
            .checked_mul(raw.dtype.clone().size_in_bytes())
            .ok_or(TensorFsError::InvalidTensorSize)?;

        if length != expected_len {
            return Err(TensorFsError::InvalidTensorSize);
        }

        let offset = data_section_start
            .checked_add(start)
            .ok_or(TensorFsError::OffsetOverflow)?;

        result.push(TensorMeta {
            name,
            dtype: raw.dtype,
            shape: raw.shape,
            offset,
            length,
        });
    }

    result.sort_by_key(|x| x.offset);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_file_from_json(json: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        let len = json.len() as u64;
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(json.as_bytes());
        bytes
    }

    #[test]
    fn parse_header_single_tensor() {
        let json = r#"
        {
            "weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16]
            }
        }
        "#;

        let bytes = make_file_from_json(json);

        let tensors = parse_header(&bytes).expect("header should parse");

        assert_eq!(tensors.len(), 1);

        let t = &tensors[0];
        assert_eq!(t.name, "weight");
        assert_eq!(t.dtype, Dtype::F32);
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.length, 16);

        let expected_offset = 8 + json.len() as u64;
        assert_eq!(t.offset, expected_offset);
    }

    #[test]
    fn parse_header_multiple_tensors() {
        let json = r#"
        {
            "tensor_a": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16]
            },
            "tensor_b": {
                "dtype": "F16",
                "shape": [4],
                "data_offsets": [16, 24]
            }
        }
        "#;

        let bytes = make_file_from_json(json);

        let tensors = parse_header(&bytes).expect("header should parse");

        assert_eq!(tensors.len(), 2);

        let data_start = 8 + json.len() as u64;

        assert_eq!(tensors[0].name, "tensor_a");
        assert_eq!(tensors[0].dtype, Dtype::F32);
        assert_eq!(tensors[0].shape, vec![2, 2]);
        assert_eq!(tensors[0].offset, data_start);
        assert_eq!(tensors[0].length, 16);

        assert_eq!(tensors[1].name, "tensor_b");
        assert_eq!(tensors[1].dtype, Dtype::F16);
        assert_eq!(tensors[1].shape, vec![4]);
        assert_eq!(tensors[1].offset, data_start + 16);
        assert_eq!(tensors[1].length, 8);
    }

    #[test]
    fn parse_header_ignores_metadata() {
        let json = r#"
        {
            "__metadata__": {
                "format": "pt",
                "model": "tiny"
            },
            "embedding.weight": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 8]
            }
        }
        "#;

        let bytes = make_file_from_json(json);

        let tensors = parse_header(&bytes).expect("header should parse");

        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "embedding.weight");
        assert_eq!(tensors[0].dtype, Dtype::F32);
        assert_eq!(tensors[0].shape, vec![2]);
        assert_eq!(tensors[0].length, 8);
    }

    #[test]
    fn parse_header_fails_when_too_short_for_len_prefix() {
        let bytes = vec![1, 2, 3, 4, 5, 6, 7];

        let err = parse_header(&bytes).expect_err("should fail");

        assert!(matches!(err, TensorFsError::TooShortForHeaderLen));
    }

    #[test]
    fn parse_header_fails_when_header_is_incomplete() {
        let json = r#"
        {
            "weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16]
            }
        }
        "#;

        let full = make_file_from_json(json);
        let truncated = full[..full.len() - 5].to_vec();

        let err = parse_header(&truncated).expect_err("should fail");

        assert!(matches!(err, TensorFsError::IncompleteHeader));
    }

    #[test]
    fn parse_header_fails_on_invalid_json() {
        let json = r#"{ "weight": { "dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16] "#;

        let bytes = make_file_from_json(json);

        let err = parse_header(&bytes).expect_err("should fail");

        assert!(matches!(err, TensorFsError::InvalidHeaderJson));
    }

    #[test]
    fn parse_header_fails_when_end_less_than_start() {
        let json = r#"
        {
            "weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [16, 0]
            }
        }
        "#;

        let bytes = make_file_from_json(json);

        let err = parse_header(&bytes).expect_err("should fail");

        assert!(matches!(err, TensorFsError::InvalidOffsets));
    }

    #[test]
    fn parse_header_fails_when_tensor_size_does_not_match_shape_and_dtype() {
        let json = r#"
        {
            "weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 8]
            }
        }
        "#;

        let bytes = make_file_from_json(json);

        let err = parse_header(&bytes).expect_err("should fail");

        assert!(matches!(err, TensorFsError::InvalidTensorSize));
    }

    #[test]
    fn parse_header_fails_on_unknown_dtype() {
        let json = r#"
        {
            "weight": {
                "dtype": "F64",
                "shape": [2],
                "data_offsets": [0, 16]
            }
        }
        "#;

        let bytes = make_file_from_json(json);

        let err = parse_header(&bytes).expect_err("should fail");

        assert!(matches!(err, TensorFsError::InvalidHeaderJson));
    }

    #[test]
    fn parse_header_allows_empty_shape_scalar_tensor_if_size_matches() {
        let json = r#"
        {
            "scalar": {
                "dtype": "I32",
                "shape": [],
                "data_offsets": [0, 4]
            }
        }
        "#;

        let bytes = make_file_from_json(json);

        let tensors = parse_header(&bytes).expect("header should parse");

        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "scalar");
        assert_eq!(tensors[0].dtype, Dtype::I32);
        assert_eq!(tensors[0].shape, Vec::<u64>::new());
        assert_eq!(tensors[0].length, 4);
    }

    #[test]
    fn parse_header_sorts_tensors_by_offset() {
        let json = r#"
    {
        "tensor_z": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [8, 16]
        },
        "tensor_a": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [0, 8]
        },
        "tensor_m": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [16, 24]
        }
    }
    "#;

        let bytes = make_file_from_json(json);

        let tensors = parse_header(&bytes).expect("header should parse");

        assert_eq!(tensors.len(), 3);

        assert_eq!(tensors[0].name, "tensor_a");
        assert_eq!(tensors[0].offset, 8 + json.len() as u64 + 0);

        assert_eq!(tensors[1].name, "tensor_z");
        assert_eq!(tensors[1].offset, 8 + json.len() as u64 + 8);

        assert_eq!(tensors[2].name, "tensor_m");
        assert_eq!(tensors[2].offset, 8 + json.len() as u64 + 16);
    }
}
