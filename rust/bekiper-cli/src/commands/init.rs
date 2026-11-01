use std::fs;
use std::path::PathBuf;

use bekiper_core::identity::DEFAULT_CONFIG_FILENAME;
use clap::Args;

// ── Project scaffold defaults ────────────────────────────────────────────
// These are user-facing filesystem conventions. They are hardcoded because
// the rest of the pipeline expects them literally (e.g. the config points
// at `models/` and `output/` by relative path). Named for documentation.

/// Canonical filename for the per-project TOML config. Hardcoded because
/// `bekiper run` defaults to reading this exact name, and the ingress
/// layer resolves it relative to the config's own parent directory.
const CONFIG_FILENAME: &str = DEFAULT_CONFIG_FILENAME;

/// Fallback project name when the directory name cannot be derived.
const DEFAULT_PROJECT_NAME: &str = "my_project";

/// Subdirectories created by `bekiper init`. The config template references
/// both by relative path (`models/...onnx`, `output_dir = "output"`), so
/// changing these would require a matching edit to the template.
const PROJECT_SUBDIRS: &[&str] = &["models", "output"];

#[derive(Args)]
pub struct InitArgs {
    /// Directory to create (default: current directory).
    #[arg(default_value = ".")]
    path: PathBuf,

    /// Project name (default: directory name).
    #[arg(short, long)]
    name: Option<String>,
}

/// Embedded config template. Placeholders: `{{project_name}}`.
const CONFIG_TEMPLATE: &str = r##"# ╔══════════════════════════════════════════════════════════════════╗
# ║  Bekiper Project Configuration                                   ║
# ║                                                                  ║
# ║  All commented-out settings show their default values.           ║
# ║  Uncomment and change only what you need.                        ║
# ║                                                                  ║
# ║  Config precedence (highest wins):                               ║
# ║    CLI flag  >  env var  >  this file  >  built-in default       ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── Project ─────────────────────────────────────────────────────────
[project]
project_name = "{{project_name}}"
data_directory = "."                    # root for relative video paths
output_dir = "output"                   # where results are written

# Runtime settings (optional).
# [project.runtime]
# minimum_seconds_tolerance = 0.5      # min bout duration to count (s)
# maximum_seconds_distraction = 0.333  # max gap to bridge between bouts (s)
# num_threads = 4                      # Rayon thread count (omit = all cores)
# memory_scheduler_enabled = true      # throttle new parallel work under RAM pressure
# max_ram_usage_fraction = 0.80        # whole-machine RAM fraction cap for new work

# ── Pipeline ────────────────────────────────────────────────────────
[pipeline]
pose_label_model = "superanimal_topviewmouse" # audit label set: "superanimal_topviewmouse" or "imen_dlc"
likelihood_threshold = 0.9             # min pose confidence to keep a keypoint
smoothing_window = 5                   # median filter window (frames)
smoothing_method = "median"            # "median" (default) or "kalman"
kalman_process_noise = 1.0             # px-scale acceleration noise
kalman_measurement_noise = 4.0         # px-scale baseline observation noise
kalman_min_likelihood = 0.05           # lower bound for confidence weighting
kalman_max_gap_frames = 30             # longer gaps remain null
max_velocity = 50.0                    # outlier rejection: max px/frame

[pipeline.quality]
max_low_likelihood_fraction = 0.05     # warning threshold for low-confidence labels
max_null_gap_seconds = 0.5             # warning threshold for invalid pose gaps
max_augmented_jump_count = 0           # residual high-velocity jumps allowed after augment

# ── Pose model ──────────────────────────────────────────────────────
[pose_model]
path = "models/superanimal_topviewmouse_hrnet_w32.onnx"
spec = "models/superanimal_topviewmouse_hrnet_w32.json"
batch_size = 32                        # frames per inference batch (0 = auto)

# ── GPU runtime ─────────────────────────────────────────────────────
# Controls HIP / MIGraphX device selection and VRAM-based batch probing.
# Env overrides: BEKIPER_GPU_DEVICE, BEKIPER_GPU_MAX_BATCH_SIZE,
# BEKIPER_GPU_VRAM_HEADROOM, BEKIPER_GPU_FALLBACK_BATCH_SIZE.
#
# [gpu]
# device_id              = 0             # HIP device index
# max_batch_size         = 1024          # hard cap on probed batch size
# vram_headroom_fraction = 0.75          # fraction of free VRAM usable by probe
# fallback_batch_size    = 32            # batch size when probe fails / CPU mode

# ── Python detector ─────────────────────────────────────────────────
# Consumed by `bekiper detect` (bekiper-auto-detect). The Rust analyzer
# ignores this section. Env overrides: BEKIPER_DEVICE, BEKIPER_SAM2_MODEL,
# BEKIPER_GROUNDING_DINO_MODEL, BEKIPER_YOLOE_MODEL, BEKIPER_DETECT_*.
#
# [detect]
# device               = "auto"          # "auto" | "cuda" | "mps" | "cpu"
# sam2_model           = "sam2.1_l.pt"
# grounding_dino_model = "IDEA-Research/grounding-dino-base"
# yoloe_model          = "yoloe-11l-seg.pt"
# confidence           = 0.25            # YOLOE / SAM2 detection confidence
# box_threshold        = 0.25            # GroundingDINO box-score threshold
# floor_percentile     = 75.0            # classical-CV background floor (P75)
# threshold_factor     = 0.85            # binary threshold = floor * this
# close_kernel_frac    = 0.007           # morph close kernel / image diagonal
# dilate_frac          = 0.03            # mouse-mask dilate / image diagonal
# min_area_frac        = 0.005           # min object area / image area
# max_area_frac        = 0.05            # max object area / image area
# mouse_min_frac       = 0.02            # min mouse blob / image area

# ── Plot theme ──────────────────────────────────────────────────────
# Consumed by `bekiper inspect plot|video`. Env overrides: BEKIPER_PLOT_DPI,
# BEKIPER_PLOT_FORMAT, BEKIPER_PLOT_FFMPEG_PRESET, and 15 others.
#
# [inspect.plot]
# color_mode           = "dark"          # dark (night mode) | light; BEKIPER_PLOT_COLOR_MODE
# min_dpi              = 300             # minimum DPI for saved figures
# fig_format           = ".svgz"         # .svgz | .jpg | .png | .pdf
# ffmpeg_preset        = "slower"        # ffmpeg encoder preset
# linewidth            = 3.0             # trajectory line width
# fontsize             = 10              # body-text font size
# title_fontsize       = 13              # in-figure titles
# annotation_fontsize  = 9               # dimension labels on scale plots
# bbox_alpha           = 0.7             # alpha of annotation bbox background
# quiver_alpha         = 0.6             # alpha of quiver arrows on video
# timeline_alpha       = 0.4             # alpha of heuristic timeline spans
# palette              = "cool"          # seaborn palette
# true_colormap        = "cool"          # matplotlib colormap for true state
# false_colormap       = "Wistia"        # matplotlib colormap for false state
# bar_primary_color    = "steelblue"     # heuristic summary bar color
# bar_secondary_color  = "darkorange"    # heuristic duration bar color
# valid_scatter_color  = "dodgerblue"    # "valid" scatter points
# invalid_scatter_color = "crimson"      # "invalid" scatter points
# video_point_color    = "cyan"          # default keypoint overlay
# video_true_color     = "lime"          # boolean-true frame marker
# video_false_color    = "red"           # boolean-false frame marker
# video_perimeter_color = "b"            # default perimeter overlay
# hint_text_color      = ""              # soft hint text; empty = the mode's muted ink

# ── Shared task defaults ────────────────────────────────────────────
# Apply to every scan. Override per-scan under [scan.task_params].
#
# freezing_window_seconds  = 1.0       # rolling window for freezing detection (s)
# freezing_displacement_threshold = 0.005  # max displacement in window to be frozen (m)
# min_dwell_seconds       = 1.0        # min time in a zone to count as entry (s)
# peripheral_band_fraction = 0.2       # peripheral zone width / arena side
#
# Body-part columns (DLC SuperAnimal defaults):
# centroid_coords = { x = "mouse_center_x", y = "mouse_center_y" }
# nose_coords     = { x = "nose_x",         y = "nose_y" }
# head_coords     = { x = "center_ear_x",   y = "center_ear_y" }
# tail_coords     = { x = "tail_base_x",    y = "tail_base_y" }

# ════════════════════════════════════════════════════════════════════
#  SCANS — each [[scan]] discovers videos in a directory
# ════════════════════════════════════════════════════════════════════

# ── NORT (Novel Object Recognition Test) ───────────────────────────
# T1 = familiarisation (identical objects)
# T2 = test (novel vs familiar)
#
# [[scan]]
# video_dir = "NORT videos"
# task_type = "object_recognition"
# meters_per_pixel = { method = "arena_side", meters = 0.40 }
# backend = ["classical_cv", "sam2_auto"]
# annotations = { dir = "annotations/nort" }
#
# [scan.recipe_rules]
# "**/T1/**" = "nort_training"
# "**/T2/**" = "nort_novelty"
#
# Object exploration thresholds:
# [scan.task_params.exploration]
# nose_max_distance        = 0.03      # max nose-to-object distance (m)
# contact_distance         = 0.005     # below this, skip angle check (m)
# max_angle_deg            = 30.0      # max head-to-object angle (degrees)
# max_nose_speed           = 0.05      # max nose speed during exploration (m/s)
# min_bout_seconds         = 1.0       # min exploration bout duration (s)
# max_gap_seconds          = 0.5       # max gap to bridge between bouts (s)
# min_total_exploration    = 15.0      # exclude animal below this total (s)
# analysis_window_seconds  = 120.0     # analysis window from first approach (s)
# perimeter_erosion_meters = 0.01      # erode masks to reduce edge artifacts (m)

# ── OFT (Open Field Test) ──────────────────────────────────────────
#
# [[scan]]
# video_dir = "OFT videos"
# task_type = "open_field"
# meters_per_pixel = { method = "arena_side", meters = 0.40 }
# recipe = "oft"
# backend = "classical_cv"
#
# Rearing — body compacts, animal stationary, >=0.5 s:
# [scan.task_params.rearing]
# max_speed             = 0.01         # max centroid speed (m/s)
# max_body_length_ratio = 0.7          # max nose-to-tail / baseline
# max_body_length       = 0.05         # absolute max nose-to-tail (m)
# forepaw_proximity     = 0.03         # forepaw-to-head threshold (m)
#
# Grooming — very slow, curled posture, >=1.0 s:
# [scan.task_params.grooming]
# max_speed         = 0.005            # max centroid speed (m/s)
# max_body_length   = 0.08             # max nose-to-tail (m)
# min_body_length   = 0.04             # min nose-to-tail (m)
# forepaw_proximity = 0.02             # forepaw-to-nose threshold (m)

# ── Y-maze ──────────────────────────────────────────────────────────
#
# [[scan]]
# video_dir = "Y-maze videos"
# task_type = "y_maze"
# meters_per_pixel = { method = "arm_length", meters = 0.35 }
# recipe = "ymaze"
# backend = ["classical_cv", "sam2_auto"]
# annotations = { dir = "annotations/ymaze" }
#
# Zone boundary thresholds:
# [scan.task_params.ymaze]
# proximal_threshold = 0.25            # fraction of arm length for proximal zone
"##;

pub fn run(args: InitArgs) -> anyhow::Result<()> {
    let dir = &args.path;

    // Create the project directory if it doesn't exist.
    if dir.as_os_str() != "." {
        fs::create_dir_all(dir)?;
    }

    let config_path = dir.join(CONFIG_FILENAME);
    if config_path.exists() {
        anyhow::bail!(
            "config already exists: {}. Use --force or remove it first.",
            config_path.display()
        );
    }

    // Derive project name.
    let project_name = args.name.unwrap_or_else(|| {
        dir.canonicalize()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_else(|| DEFAULT_PROJECT_NAME.into())
    });

    let config = CONFIG_TEMPLATE.replace("{{project_name}}", &project_name);
    fs::write(&config_path, config)?;

    // Create conventional subdirectories.
    for sub in PROJECT_SUBDIRS {
        let p = dir.join(sub);
        if !p.exists() {
            fs::create_dir_all(&p)?;
        }
    }

    println!("Initialized bekiper project in {}", dir.display());
    println!("  config:  {}", config_path.display());
    println!("  output:  {}", dir.join("output").display());
    println!();
    println!("Next steps:");
    println!("  1. Place your pose model in models/");
    println!("  2. Edit bekiper_config.toml — uncomment the scans you need");
    println!("  3. Run: bekiper run");

    Ok(())
}
