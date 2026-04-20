use core::error::TensorFsError;

pub enum ModelRef {
    HuggingFace { model_id: String },
}

impl std::str::FromStr for ModelRef {
    type Err = TensorFsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(rest) = s.strip_prefix("hf://") {
            if rest.is_empty() {
                return Err(TensorFsError::InvalidArgument);
            }

            return Ok(ModelRef::HuggingFace {
                model_id: rest.to_string(),
            });
        }

        Err(TensorFsError::InvalidArgument)
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::ModelRef;

    #[test]
    fn parses_valid_hf_model_ref() {
        let model_ref = ModelRef::from_str("hf://Qwen/Qwen2.5-0.5B").unwrap();

        match model_ref {
            ModelRef::HuggingFace { model_id } => {
                assert_eq!(model_id, "Qwen/Qwen2.5-0.5B");
            }
        }
    }

    #[test]
    fn rejects_sources_without_hf_model_ref() {
        for source in ["Qwen/Qwen2.5-0.5B", "hf://"] {
            assert!(
                ModelRef::from_str(source).is_err(),
                "{source} should be invalid"
            );
        }
    }
}
