/// Overlay annotation perimeters onto a video frame and save as PNG.
///
/// Extracts the middle frame from `video_path`, loads perimeters from
/// `annotation_dir` (optionally filtered by `perimeter_id`), draws them
/// onto the frame, and writes the result to `output_path`.
pub fn overlay_annotation(
    video_path: &Path,
    annotation_dir: &Path,
    output_path: &Path,
) -> Result<()> {
    use crate::overlay;
    use bekiper_annotation::{AnnotationResolver, extract_video_context};

    // Try to resolve the correct perimeter via metadata + video context
    let resolver = AnnotationResolver::from_dir(annotation_dir).ok();
    let ctx = extract_video_context(video_path);

    let perimeters = match (&resolver, &ctx) {
        (Some(resolver), Some(ctx)) => match resolver.resolve_with_fallback(ctx)? {
            Some((specs, fallback)) => {
                if fallback {
                    tracing::info!(
                        video = %video_path.display(),
                        "using fallback (shared) perimeter"
                    );
                }
                specs
            }
            None => {
                tracing::warn!(
                    video = %video_path.display(),
                    animal = ctx.animal,
                    stage = ctx.stage,
                    phase_part = ctx.phase_part,
                    "no perimeter mapping found for video"
                );
                return Err(BekiperError::Data(format!(
                    "no perimeter mapping for {} (animal={}, stage={}, phase_part={})",
                    video_path.display(),
                    ctx.animal,
                    ctx.stage,
                    ctx.phase_part
                )));
            }
        },
        _ => {
            // No resolver or no context — fall back to sole perimeter
            let plugin_dir = annotation_dir.join("plugin_files");
            let mut all = Vec::new();
            if let Ok(specs) = bekiper_annotation::parse_radial_maze(&plugin_dir) {
                all.extend(specs);
            }
            for entry in std::fs::read_dir(&plugin_dir)
                .into_iter()
                .flatten()
                .flatten()
            {
                let p = entry.path();
                let name = p.file_name().unwrap_or_default().to_string_lossy();
                if !name.starts_with("perimeter-rectangle-") {
                    continue;
                }
                match p.extension().and_then(|e| e.to_str()) {
                    Some("csv") => {
                        if let Ok(specs) = bekiper_annotation::parse_rectangle_csv(&p) {
                            all.extend(specs);
                        }
                    }
                    Some("json") => {
                        if let Ok(specs) = bekiper_annotation::parse_coco_json(&p) {
                            all.extend(specs);
                        }
                    }
                    _ => {}
                }
            }
            all
        }
    };

    if perimeters.is_empty() {
        return Err(BekiperError::Data(format!(
            "no perimeters found in {}",
            annotation_dir.display()
        )));
    }

    // Extract middle frame from video
    let mut decoder = VideoDecoder::open(video_path)?;
    let total = decoder.info().total_frames;
    let mid = total / 2;
    if mid > 0 {
        decoder.seek_to(mid)?;
    }
    let batch = decoder.next_batch(1)?;
    let (_, frame_img) = batch
        .into_iter()
        .next()
        .ok_or_else(|| BekiperError::Data("failed to decode a frame from video".into()))?;

    let mut rgb = overlay::to_rgb(&frame_img);
    overlay::draw_perimeters(&mut rgb, &perimeters);

    image::DynamicImage::ImageRgb8(rgb)
        .save(output_path)
        .map_err(|e| BekiperError::Data(format!("save overlay: {e}")))?;

    tracing::info!(
        video = %video_path.display(),
        annotations = %annotation_dir.display(),
        output = %output_path.display(),
        n_perimeters = perimeters.len(),
        "overlay annotation saved"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 1a: Detect perimeters from video frame
// ---------------------------------------------------------------------------
