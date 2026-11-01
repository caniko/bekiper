#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ImenCanonicalContextKey {
    animal: u32,
    stage: u32,
    phase_part: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ImenVideoLookupKey {
    video_number: u32,
    phase: String,
    phase_part: Option<u32>,
}

#[derive(Debug, Clone)]
struct ImenVideoMetadataCandidate {
    phase_part: u32,
    stage: u32,
    animal: u32,
    source_row: usize,
    context: IngressSessionContext,
}

#[derive(Debug, Default)]
struct ImenMetadataContexts {
    by_animal_stage_part: HashMap<(u32, u32, u32), IngressSessionContext>,
    by_video_phase: HashMap<(u32, String), Vec<ImenVideoMetadataCandidate>>,
    canonical_contexts: std::collections::HashSet<ImenCanonicalContextKey>,
}

impl ImenMetadataContexts {
    fn context_for_oft_video(&self, video_path: &Path) -> Option<IngressSessionContext> {
        let animal = bekiper_annotation::extract_video_number(video_path)?;
        let part = imen_video_phase_part(video_path)?;
        self.by_animal_stage_part.get(&(animal, 0, part)).cloned()
    }

    fn context_for_video(&self, video_path: &Path) -> Option<IngressSessionContext> {
        if let Some(key) = imen_video_lookup_key(video_path)
            && let Some(context) = self.best_context_for_lookup(&key)
        {
            return Some(context);
        }

        bekiper_annotation::extract_video_context(video_path).and_then(|ctx| {
            self.by_animal_stage_part
                .get(&(ctx.animal, ctx.stage, ctx.phase_part))
                .cloned()
        })
    }

    fn context_for_video_recipe(
        &self,
        video_path: &Path,
        recipe: DetectionRecipe,
    ) -> Option<IngressSessionContext> {
        let key = imen_video_lookup_key(video_path)?;
        let stage = match recipe {
            DetectionRecipe::NortTraining => 1,
            DetectionRecipe::NortNovelty => 2,
            _ => return None,
        };
        let phase_part = key.phase_part?;
        if let Some(context) = self
            .by_animal_stage_part
            .get(&(key.video_number, stage, phase_part))
        {
            return Some(context.clone());
        }
        for candidate in self
            .candidates_for_lookup(&key)
            .into_iter()
            .filter(|candidate| candidate.stage != 0)
        {
            if let Some(context) =
                self.by_animal_stage_part
                    .get(&(candidate.animal, stage, phase_part))
            {
                return Some(context.clone());
            }
        }
        self.by_animal_stage_part
            .get(&(key.video_number, stage, phase_part))
            .cloned()
    }

    fn best_context_for_lookup(&self, key: &ImenVideoLookupKey) -> Option<IngressSessionContext> {
        let candidates = self.candidates_for_lookup(key);
        let selected = candidates
            .iter()
            .copied()
            .find(|candidate| candidate.stage != 0 && self.has_canonical_session(candidate))
            .or_else(|| {
                candidates
                    .iter()
                    .copied()
                    .find(|candidate| candidate.stage != 0)
            })
            .or_else(|| candidates.first().copied())?;
        Some(selected.context.clone())
    }

    fn candidates_for_lookup(&self, key: &ImenVideoLookupKey) -> Vec<&ImenVideoMetadataCandidate> {
        self.by_video_phase
            .get(&(key.video_number, key.phase.clone()))
            .into_iter()
            .flat_map(|candidates| candidates.iter())
            .filter(|candidate| {
                key.phase_part
                    .map(|phase_part| candidate.phase_part == phase_part)
                    .unwrap_or(true)
            })
            .collect()
    }

    fn candidates_for_video_phase(
        &self,
        key: &ImenVideoLookupKey,
    ) -> Vec<&ImenVideoMetadataCandidate> {
        self.by_video_phase
            .get(&(key.video_number, key.phase.clone()))
            .into_iter()
            .flat_map(|candidates| candidates.iter())
            .collect()
    }

    fn has_stage_zero_lookup_candidate(&self, video_path: &Path) -> bool {
        let Some(key) = imen_video_lookup_key(video_path) else {
            return false;
        };
        self.candidates_for_lookup(&key)
            .iter()
            .any(|candidate| candidate.stage == 0)
    }

    fn candidate_summary_for_video(&self, video_path: &Path) -> Option<String> {
        let key = imen_video_lookup_key(video_path)?;
        let candidates = self.candidates_for_video_phase(&key);
        if candidates.is_empty() {
            return Some("none".into());
        }
        Some(format_candidate_summary(candidates))
    }

    fn has_canonical_session(&self, candidate: &ImenVideoMetadataCandidate) -> bool {
        self.canonical_contexts.contains(&ImenCanonicalContextKey {
            animal: candidate.animal,
            stage: candidate.stage,
            phase_part: candidate.phase_part,
        })
    }
}

fn load_imen_contexts(annotation_dir: &Path) -> Result<ImenMetadataContexts> {
    let metadata_path = annotation_dir.join("metadata.csv");
    if !metadata_path.exists() {
        return Ok(ImenMetadataContexts::default());
    }

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(&metadata_path)
        .map_err(|e| {
            BekiperError::Data(format!("failed to read {}: {e}", metadata_path.display()))
        })?;

    let headers = reader
        .headers()
        .map_err(|e| BekiperError::Data(format!("{}: {e}", metadata_path.display())))?
        .clone();

    let mut current_phase = String::new();
    let mut current_phase_part = String::new();
    let mut contexts = ImenMetadataContexts {
        canonical_contexts: load_canonical_context_keys(annotation_dir)?,
        ..Default::default()
    };

    for (row_index, record) in reader.records().enumerate() {
        let record =
            record.map_err(|e| BekiperError::Data(format!("{}: {e}", metadata_path.display())))?;

        if let Some(phase) = get(&headers, &record, "Phase").filter(|s| !s.is_empty()) {
            current_phase = phase.to_string();
        }
        if let Some(part) = get(&headers, &record, "PhasePart")
            .or_else(|| get(&headers, &record, "Part"))
            .filter(|s| !s.is_empty())
        {
            current_phase_part = part.to_string();
        }

        let Some(animal) = get(&headers, &record, "Animal")
            .or_else(|| get(&headers, &record, "Number"))
            .and_then(|s| s.parse::<u32>().ok())
        else {
            continue;
        };
        let stage = get(&headers, &record, "Stage")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);
        let phase_part = current_phase_part.parse::<u32>().unwrap_or(0);
        let source_row = row_index + 2;

        let context = IngressSessionContext {
            profile: IngressProfileId::VideoMap.as_str().into(),
            source_file: Some(metadata_path.display().to_string()),
            source_row: Some(source_row),
            animal: Some(animal.to_string()),
            trial: get(&headers, &record, "TrialNumber").map(ToOwned::to_owned),
            phase: (!current_phase.is_empty()).then(|| current_phase.clone()),
            phase_part: (!current_phase_part.is_empty()).then(|| current_phase_part.clone()),
            stage: Some(stage.to_string()),
            apparatus: get(&headers, &record, "Apparatus").map(ToOwned::to_owned),
            treatment: get(&headers, &record, "Treatment").map(ToOwned::to_owned),
            group: get(&headers, &record, "Group").map(ToOwned::to_owned),
            sex: get(&headers, &record, "Sex").map(ToOwned::to_owned),
            age: get(&headers, &record, "Age (days)")
                .or_else(|| get(&headers, &record, "Age (months)"))
                .map(ToOwned::to_owned),
            genotype: join_nonempty([
                get(&headers, &record, "HCAR1"),
                get(&headers, &record, "5XFAD"),
            ]),
            object_a: get(&headers, &record, "Object A (D=down, U=up)").map(ToOwned::to_owned),
            object_b: get(&headers, &record, "Object B (D=down, U=up)").map(ToOwned::to_owned),
            fecal_boli_count: get(&headers, &record, "FecalBoli")
                .and_then(|s| s.parse::<u32>().ok()),
            comments: get(&headers, &record, "All comments").map(ToOwned::to_owned),
            sequence: None,
        };

        if let Some(previous) = contexts
            .by_animal_stage_part
            .insert((animal, stage, phase_part), context.clone())
        {
            return Err(BekiperError::Data(format!(
                "{}: duplicate (Animal, Stage, PhasePart) ({animal}, {stage}, {phase_part}) at rows {} and {source_row}",
                metadata_path.display(),
                previous.source_row.unwrap_or(0)
            )));
        }

        if let Some(video_number) =
            get(&headers, &record, "Video_file_name").and_then(extract_metadata_video_number)
            && !current_phase.is_empty()
        {
            contexts
                .by_video_phase
                .entry((video_number, current_phase.clone()))
                .or_default()
                .push(ImenVideoMetadataCandidate {
                    phase_part,
                    stage,
                    animal,
                    source_row,
                    context,
                });
        }
    }

    Ok(contexts)
}

pub(crate) fn missing_imen_metadata_context_message(
    annotation_dir: &Path,
    video_path: &Path,
) -> Option<String> {
    let metadata_path = annotation_dir.join("metadata.csv");
    if !metadata_path.exists() {
        return None;
    }
    let key = imen_video_lookup_key(video_path)?;
    let contexts = load_imen_contexts(annotation_dir).ok()?;
    if !contexts.candidates_for_lookup(&key).is_empty() {
        return None;
    }
    let candidates = contexts.candidates_for_video_phase(&key);
    let candidate_summary = if candidates.is_empty() {
        "none".to_string()
    } else {
        format_candidate_summary(candidates)
    };
    Some(format!(
        "no metadata.csv row matched video number {}, phase {}, phase_part {} for {}; candidate rows for video/phase: {}",
        key.video_number,
        key.phase,
        key.phase_part
            .map(|part| part.to_string())
            .unwrap_or_else(|| "unknown".into()),
        video_path.display(),
        candidate_summary
    ))
}

fn format_candidate_summary(candidates: Vec<&ImenVideoMetadataCandidate>) -> String {
    candidates
        .iter()
        .map(|candidate| {
            format!(
                "row {} phase_part {} stage {} animal {}",
                candidate.source_row, candidate.phase_part, candidate.stage, candidate.animal
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

pub(crate) fn is_imen_habituation_context(context: Option<&IngressSessionContext>) -> bool {
    let Some(context) = context else {
        return false;
    };
    context.profile == IngressProfileId::VideoMap.as_str()
        && context.stage.as_deref() == Some("0")
        && matches!(context.phase.as_deref(), Some("A" | "B"))
}

pub(crate) fn validate_imen_oft_metadata_bindings(experiments: &[ExperimentConfig]) -> Result<()> {
    for experiment in experiments {
        if !crate::analysis_exclusion::is_imen_oft_habituation_experiment(experiment)
            || experiment.annotation_dir.is_none()
        {
            continue;
        }
        if experiment.session_context.as_ref().is_none_or(|context| {
            context.stage.as_deref() != Some("0")
                || context
                    .trial
                    .as_deref()
                    .and_then(|value| value.parse::<u32>().ok())
                    .is_none()
        }) {
            return Err(BekiperError::Config(format!(
                "OFT experiment {} requires an unambiguous metadata.csv (Animal, 0, PhasePart) binding with original TrialNumber",
                experiment.name
            )));
        }
    }
    Ok(())
}

fn load_canonical_context_keys(
    annotation_dir: &Path,
) -> Result<std::collections::HashSet<ImenCanonicalContextKey>> {
    let Some(store) =
        bekiper_annotation::canonical::CanonicalAnnotationStore::open(annotation_dir)?
    else {
        return Ok(std::collections::HashSet::new());
    };
    let keys = store
        .load_sessions()?
        .into_iter()
        .filter_map(|session| {
            Some(ImenCanonicalContextKey {
                animal: session.animal?,
                stage: session.stage?,
                phase_part: session.phase_part?,
            })
        })
        .collect();
    Ok(keys)
}

fn get<'a>(
    headers: &csv::StringRecord,
    record: &'a csv::StringRecord,
    name: &str,
) -> Option<&'a str> {
    let idx = headers
        .iter()
        .position(|h| h.trim().trim_start_matches('\u{feff}') == name)?;
    record.get(idx).map(str::trim).filter(|s| !s.is_empty())
}

fn join_nonempty(values: [Option<&str>; 2]) -> Option<String> {
    let parts = values.into_iter().flatten().collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join("/"))
}

fn imen_video_lookup_key(video_path: &Path) -> Option<ImenVideoLookupKey> {
    let video_number = bekiper_annotation::extract_video_number(video_path)?;
    let path = video_path.to_str()?.to_lowercase();
    let phase = if path.contains("config a") || path.contains("configa") {
        "A"
    } else if path.contains("config b") || path.contains("configb") {
        "B"
    } else {
        return None;
    };
    // PhasePart is the pre/post timepoint; T1/T2 is the training/test stage.
    // Prefer the filename, then nearby directories, and match whole tokens so
    // an ancestor such as "preparation" cannot change a video's identity.
    let phase_part = imen_video_phase_part(video_path);

    Some(ImenVideoLookupKey {
        video_number,
        phase: phase.into(),
        phase_part,
    })
}

fn imen_video_phase_part(video_path: &Path) -> Option<u32> {
    video_path
        .to_str()?
        .to_lowercase()
        .split('/')
        .rev()
        .flat_map(|component| component.split(|c: char| !c.is_ascii_alphanumeric()))
        .find_map(|token| match token {
            "pre" => Some(1),
            "post" => Some(2),
            _ => None,
        })
}

fn metadata_recipe_for_context(
    context: Option<&IngressSessionContext>,
) -> Option<Option<DetectionRecipe>> {
    let context = context?;
    if context.profile != IngressProfileId::VideoMap.as_str()
        || !matches!(context.phase.as_deref(), Some("A" | "B"))
    {
        return None;
    }
    match context.stage.as_deref() {
        Some("0") => Some(None),
        Some("1") => Some(Some(DetectionRecipe::NortTraining)),
        Some("2") => Some(Some(DetectionRecipe::NortNovelty)),
        _ => None,
    }
}

fn extract_metadata_video_number(value: &str) -> Option<u32> {
    let lower = value.to_lowercase();
    let (_, after_test) = lower.split_once("test")?;
    let digits = after_test
        .trim_start()
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    digits.parse().ok()
}

fn relative_to_base(path: PathBuf, base_dir: &Path) -> PathBuf {
    path.strip_prefix(base_dir).unwrap_or(&path).to_path_buf()
}

#[cfg(test)]
#[path = "load_imen_contexts/tests.rs"]
mod tests;
