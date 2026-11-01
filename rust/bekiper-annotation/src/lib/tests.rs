#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    include!("tests/make_temp_dir.rs");
    include!("tests/radial_maze_lines_expand_to_arm_parallelograms.rs");
}
