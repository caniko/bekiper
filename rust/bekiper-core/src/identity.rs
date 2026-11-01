/// Public display name used in CLI/docs-facing messages.
pub const DISPLAY_NAME: &str = "Bekiper";

/// Lowercase project identity used in package names and generated paths.
pub const PROJECT_NAME: &str = "bekiper";

/// Canonical project configuration filename.
pub const DEFAULT_CONFIG_FILENAME: &str = "bekiper_config.toml";

/// Canonical top-level command name.
pub const CLI_BINARY_NAME: &str = "bekiper";

/// Canonical annotation editor binary name.
pub const ANNOTATE_BINARY_NAME: &str = "bekiper-annotate";

/// Environment-variable prefix for Bekiper-specific settings.
pub const ENV_PREFIX: &str = "BEKIPER";

/// Environment variable used to override annotation-editor log output.
pub const ANNOTATE_LOG_ENV: &str = "BEKIPER_ANNOTATE_LOG";

/// Default annotation-editor review queue filename.
pub const ANNOTATE_REVIEW_FILENAME: &str = ".bekiper-annotate-review.toml";

/// Default annotation-editor log filename.
pub const ANNOTATE_LOG_FILENAME: &str = "bekiper-annotate.log";
