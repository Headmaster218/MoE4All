//! Model-native reasoning levels. The model's chat template decides which subset it supports.

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        })
    }
}

impl std::str::FromStr for ReasoningEffort {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::Xhigh),
            "max" => Ok(Self::Max),
            _ => Err(format!(
                "expected low, medium, high, xhigh, or max; got {raw:?} (valid levels depend on the model)"
            )),
        }
    }
}

impl super::ConfigValue for ReasoningEffort {
    fn parse_set(raw: &str) -> Result<Self, String> {
        raw.parse()
    }

    fn to_display(&self) -> String {
        self.to_string()
    }
}
