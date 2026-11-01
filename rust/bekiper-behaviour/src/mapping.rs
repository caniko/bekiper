//! Task type registry.

use bekiper_core::types::TaskType;

/// Return a static slice of every [`TaskType`] this crate can build.
pub fn all_task_types() -> &'static [TaskType] {
    &[
        TaskType::ObjectRecognition,
        TaskType::YMaze,
        TaskType::OpenField,
        TaskType::Cheeseboard,
    ]
}
