include!("../workflow/augment/redetectedkeypoint.rs");
include!("../workflow/augment/apply_redetected_candidates_with.rs");
#[cfg(feature = "pose-gpu")]
include!("../workflow/augment/tests.rs");
