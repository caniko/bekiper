//! Manual annotation support for pre-annotated perimeters.
//!
//! Reads annotation plugin files in these formats:
//! - **Rectangle CSV**: `label,bbox_x,bbox_y,bbox_w,bbox_h,image,w,h`
//! - **COCO JSON**: segmentation polygons and/or bboxes
//! - **Arm-lines CSV** + center-triangle JSON -> RadialMaze (Y-maze)
//!
//! Annotations are resolved via a metadata CSV that maps video trial numbers
//! to perimeter IDs, or via an explicit `video_map.csv`.

pub mod canonical;
mod editable;
pub use editable::radial_arm_distal_edge;

include!("lib/canonicalize_vertices.rs");
include!("lib/parse_circle_point_csv.rs");
include!("lib/extract_video_context.rs");
include!("lib/annotationresolver.rs");
include!("lib/tests.rs");
