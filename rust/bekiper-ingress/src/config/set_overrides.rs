//! Generic `KEY=VALUE` override layer (`BEKIPER_SET` env, `bekiper run --set`).
//!
//! Applied to the parsed `toml::Value` before typed deserialization, so every
//! config key is reachable: default < TOML < `BEKIPER_SET` < `--set`.
//!
//! Key syntax: `<task>.<path>` edits the base `task_params` of every task of
//! that type (`[[scan]]`, `[[animal_scan]]`, `[[experiments]]`,
//! `[[ingress_profile]]` and `[[ingress_profile.tasks]]`); `<task>@<variant>.<path>`
//! edits the named variant; any other first segment is a root path.
//! `<path>` is relative to `task_params`, e.g. `nort.exploration.whisker.apex_fraction`
//! or `ymaze@arm25.ymaze.proximal_threshold`.
use bekiper_core::error::{BekiperError, Result};
use bekiper_core::types::TaskType;
use toml::Value;

use super::IngressConfig;

/// One applied `KEY=VALUE` override and where it landed.
#[derive(Debug, Clone, PartialEq)]
pub struct AppliedOverride {
    /// `"env"` (`BEKIPER_SET`) or `"cli"` (`--set`).
    pub layer: &'static str,
    pub key: String,
    /// Concrete TOML paths written, e.g. `ingress_profile[0].tasks[2].task_params.ymaze.proximal_threshold`.
    pub targets: Vec<String>,
    pub value: Value,
}

#[derive(Clone)]
enum Seg {
    Key(String),
    Idx(usize),
}

fn render(path: &[Seg]) -> String {
    path.iter().fold(String::new(), |mut out, seg| {
        match seg {
            Seg::Idx(i) => out.push_str(&format!("[{i}]")),
            Seg::Key(k) => {
                if !out.is_empty() {
                    out.push('.');
                }
                out.push_str(k);
            }
        }
        out
    })
}

fn cfg_err(message: String) -> BekiperError {
    BekiperError::Config(message)
}

/// Parse `KEY=VALUE`; the value is a TOML value, falling back to a bare string.
pub fn parse_pair(raw: &str) -> Result<(String, Value)> {
    let (key, value) = raw
        .split_once('=')
        .filter(|(k, _)| !k.trim().is_empty())
        .ok_or_else(|| cfg_err(format!("override `{raw}` must be KEY=VALUE")))?;
    let value = value.trim();
    let parsed = toml::from_str::<toml::Table>(&format!("v = {value}"))
        .ok()
        .and_then(|mut t| t.remove("v"))
        .unwrap_or_else(|| Value::String(value.to_string()));
    Ok((key.trim().to_string(), parsed))
}

/// `BEKIPER_SET`: `;`-separated pairs. ponytail: a `;` inside a quoted value
/// cannot be expressed here; use `--set` for those.
fn env_pairs(lookup: &impl Fn(&str) -> Option<String>) -> Vec<String> {
    lookup("BEKIPER_SET")
        .map(|v| {
            v.split(';')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

fn task_matches(table: &Value, task: TaskType) -> bool {
    table
        .get("task_type")
        .and_then(Value::as_str)
        .and_then(|s| s.parse::<TaskType>().ok())
        == Some(task)
}

/// Base paths (to a table holding task params) of every task of `task`,
/// or of the named variant when `variant` is set. A bare key (no variant) also
/// targets every variant that already sets `rest`, since variants win after merge.
fn task_targets(
    root: &Value,
    task: TaskType,
    variant: Option<&str>,
    rest: &[Seg],
) -> Vec<Vec<Seg>> {
    let mut out = Vec::new();
    let mut visit = |path: Vec<Seg>, item: &Value, base: bool| {
        if base {
            let mut p = path.clone();
            p.push(Seg::Key("task_params".into()));
            out.push(p);
        }
        for (j, v) in item
            .get("variants")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            let name = v.get("name").and_then(Value::as_str);
            if match variant {
                Some(_) => variant == name,
                None => get(v, rest).is_some(),
            } {
                let mut p = path.clone();
                p.extend([Seg::Key("variants".into()), Seg::Idx(j)]);
                out.push(p);
            }
        }
    };
    let list = |key: &str| {
        root.get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
    };
    for key in ["experiments", "scan", "animal_scan"] {
        for (i, item) in list(key).enumerate() {
            if task_matches(item, task) {
                visit(
                    vec![Seg::Key(key.into()), Seg::Idx(i)],
                    item,
                    variant.is_none(),
                );
            }
        }
    }
    for (i, profile) in list("ingress_profile").enumerate() {
        let at = vec![Seg::Key("ingress_profile".into()), Seg::Idx(i)];
        if task_matches(profile, task) && variant.is_none() {
            visit(at.clone(), profile, true);
        }
        for (j, item) in profile
            .get("tasks")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            if task_matches(item, task) {
                let mut p = at.clone();
                p.extend([Seg::Key("tasks".into()), Seg::Idx(j)]);
                visit(p, item, variant.is_none());
            }
        }
    }
    out
}

fn resolve(root: &Value, key: &str) -> Result<Vec<Vec<Seg>>> {
    let (head, rest) = key.split_once('.').unwrap_or((key, ""));
    let (task_name, variant) = match head.split_once('@') {
        Some((t, v)) => (t, Some(v)),
        None => (head, None),
    };
    let rest_segs: Vec<Seg> = rest
        .split('.')
        .filter(|s| !s.is_empty())
        .map(|s| Seg::Key(s.into()))
        .collect();
    let Ok(task) = task_name.parse::<TaskType>() else {
        if variant.is_some() {
            return Err(cfg_err(format!(
                "override `{key}`: `{task_name}` is not a task type"
            )));
        }
        return Ok(vec![key.split('.').map(|s| Seg::Key(s.into())).collect()]);
    };
    if rest.is_empty() {
        return Err(cfg_err(format!(
            "override `{key}` names a task but no parameter path"
        )));
    }
    let bases = task_targets(root, task, variant, &rest_segs);
    if bases.is_empty() {
        return Err(cfg_err(match variant {
            Some(v) => format!("override `{key}`: no `{task_name}` task has a variant named `{v}`"),
            None => format!("override `{key}`: config has no `{task_name}` task"),
        }));
    }
    Ok(bases
        .into_iter()
        .map(|mut p| {
            p.extend(rest_segs.iter().cloned());
            p
        })
        .collect())
}

fn set(root: &mut Value, path: &[Seg], value: Value, key: &str) -> Result<()> {
    let mut cur = root;
    for seg in path {
        cur = match seg {
            Seg::Idx(i) => cur.get_mut(*i),
            Seg::Key(k) => cur.as_table_mut().map(|t| {
                t.entry(k.clone())
                    .or_insert_with(|| Value::Table(Default::default()))
            }),
        }
        .ok_or_else(|| {
            cfg_err(format!(
                "override `{key}`: `{}` is not a table",
                render(path)
            ))
        })?;
    }
    *cur = value;
    Ok(())
}

fn get<'a>(root: &'a Value, path: &[Seg]) -> Option<&'a Value> {
    path.iter().try_fold(root, |cur, seg| match seg {
        Seg::Idx(i) => cur.get(*i),
        Seg::Key(k) => cur.get(k.as_str()),
    })
}

/// Number-tolerant subset match: serde may re-emit ints as floats and add defaults.
fn retained(got: &Value, want: &Value) -> bool {
    match (got, want) {
        (Value::Table(g), Value::Table(w)) => w
            .iter()
            .all(|(k, v)| g.get(k).is_some_and(|x| retained(x, v))),
        (Value::Array(g), Value::Array(w)) => {
            g.len() == w.len() && g.iter().zip(w).all(|(a, b)| retained(a, b))
        }
        (Value::Integer(i), Value::Float(f)) | (Value::Float(f), Value::Integer(i)) => {
            *i as f64 == *f
        }
        _ => got == want,
    }
}

/// Typed config as TOML-shaped data. JSON is the carrier because `toml::Value::try_from`
/// rejects unit fields and top-level `None`; nulls are dropped like TOML would.
fn from_json(j: &serde_json::Value) -> Value {
    use serde_json::Value as J;
    match j {
        J::Null => Value::Table(Default::default()),
        J::Bool(b) => Value::Boolean(*b),
        J::Number(n) => n.as_i64().map_or_else(
            || Value::Float(n.as_f64().unwrap_or(f64::NAN)),
            Value::Integer,
        ),
        J::String(s) => Value::String(s.clone()),
        J::Array(a) => Value::Array(a.iter().filter(|v| !v.is_null()).map(from_json).collect()),
        J::Object(o) => Value::Table(
            o.iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k.clone(), from_json(v)))
                .collect(),
        ),
    }
}

fn leaves(value: &Value, path: &mut Vec<Seg>, out: &mut Vec<String>) {
    let mut recurse = |seg: Seg, v: &Value| {
        path.push(seg);
        leaves(v, path, out);
        path.pop();
    };
    match value {
        Value::Table(t) => t.iter().for_each(|(k, v)| recurse(Seg::Key(k.clone()), v)),
        Value::Array(a) if a.iter().any(|v| v.is_table() || v.is_array()) => a
            .iter()
            .enumerate()
            .for_each(|(i, v)| recurse(Seg::Idx(i), v)),
        _ => out.push(render(path)),
    }
}

/// Inverse of [`render`] for the targets recorded in an [`AppliedOverride`].
fn parse_target(target: &str) -> Vec<Seg> {
    let mut out = Vec::new();
    for part in target.split('.') {
        let (key, indices) = part.split_once('[').map_or((part, ""), |(k, i)| (k, i));
        if !key.is_empty() {
            out.push(Seg::Key(key.into()));
        }
        for index in indices.split('[').filter(|i| !i.is_empty()) {
            if let Ok(i) = index.trim_end_matches(']').parse() {
                out.push(Seg::Idx(i));
            }
        }
    }
    out
}

/// Apply `BEKIPER_SET` then `cli_set` to a raw config value (key syntax in the module
/// docs). Pair with [`ensure_overrides_retained`] so unknown keys fail.
pub fn apply_set_overrides(
    root: &mut Value,
    cli_set: &[String],
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<Vec<AppliedOverride>> {
    let env = env_pairs(lookup);
    let mut record = Vec::new();
    let layers = env
        .iter()
        .map(|s| ("env", s))
        .chain(cli_set.iter().map(|s| ("cli", s)));
    for (layer, raw) in layers {
        let (key, value) = parse_pair(raw)?;
        let targets = resolve(root, &key)?;
        for path in &targets {
            set(root, path, value.clone(), &key)?;
        }
        record.push(AppliedOverride {
            layer,
            targets: targets.iter().map(|p| render(p)).collect(),
            key,
            value,
        });
    }
    Ok(record)
}

/// Require every override's last write to survive into `typed`, the deserialized
/// config re-serialized (unknown or ignored keys fail).
pub fn ensure_overrides_retained(
    typed: &serde_json::Value,
    record: &[AppliedOverride],
) -> Result<()> {
    let typed = from_json(typed);
    // Later layers win, so only the last write to each path must survive.
    for (i, rec) in record.iter().enumerate() {
        for target in &rec.targets {
            let overridden_later = record[i + 1..].iter().any(|r| r.targets.contains(target));
            if !overridden_later
                && !get(&typed, &parse_target(target)).is_some_and(|got| retained(got, &rec.value))
            {
                return Err(cfg_err(format!(
                    "unknown or ignored override key `{}` (`{target}` is not a config key)",
                    rec.key
                )));
            }
        }
    }
    Ok(())
}

impl IngressConfig {
    /// Parse `content`, apply `BEKIPER_SET` then `cli_set` to the TOML value, deserialize,
    /// and require every overridden path to survive into the typed config (unknown keys fail).
    pub(super) fn parse_with_set_overrides(
        content: &str,
        cli_set: &[String],
        lookup: &impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let env = env_pairs(lookup);
        if env.is_empty() && cli_set.is_empty() {
            let mut config: Self = toml::from_str(content).map_err(|e| cfg_err(e.to_string()))?;
            config.toml_paths = Self::toml_leaf_paths(content);
            return Ok(config);
        }
        let mut root: Value = toml::from_str(content).map_err(|e| cfg_err(e.to_string()))?;
        let toml_paths = Self::toml_leaf_paths(content);
        let record = apply_set_overrides(&mut root, cli_set, lookup)?;
        let mut config: Self = root.try_into().map_err(|e| cfg_err(e.to_string()))?;
        let typed = serde_json::to_value(&config).map_err(|e| cfg_err(e.to_string()))?;
        ensure_overrides_retained(&typed, &record)?;
        config.override_record = record;
        config.toml_paths = toml_paths;
        Ok(config)
    }

    /// Re-apply `--set` pairs to the typed config after the dedicated env overlays,
    /// so CLI beats env. ponytail: round-trips the whole config through TOML-shaped
    /// data; only runs when `--set` is given (`typed_roundtrip_is_lossless_for_imen` guards it).
    pub(super) fn reapply_cli_sets(&mut self, cli_set: &[String]) -> Result<()> {
        let mut root = serde_json::to_value(&*self)
            .map(|j| from_json(&j))
            .map_err(|e| cfg_err(e.to_string()))?;
        for raw in cli_set {
            let (key, value) = parse_pair(raw)?;
            for path in resolve(&root, &key)? {
                set(&mut root, &path, value.clone(), &key)?;
            }
        }
        let mut next: Self = root.try_into().map_err(|e| cfg_err(e.to_string()))?;
        next.override_record = std::mem::take(&mut self.override_record);
        next.toml_paths = std::mem::take(&mut self.toml_paths);
        *self = next;
        Ok(())
    }

    /// Record a dedicated shorthand (`--ymaze-dwell-seconds`, `BEKIPER_NORT_WHISKER_*`, ...)
    /// as the generic key it set. Skipped when no task of that type exists.
    pub(super) fn record_shorthand(&mut self, layer: &'static str, key: &str, value: Value) {
        let Ok(typed) = serde_json::to_value(&*self).map(|j| from_json(&j)) else {
            return; // coverage: unreachable — IngressConfig serializes to JSON infallibly (no map keys or non-finite floats)
        };
        let targets: Vec<String> = resolve(&typed, key)
            .unwrap_or_default()
            .iter()
            .map(|p| render(p))
            .collect();
        if !targets.is_empty() {
            self.override_record.push(AppliedOverride {
                layer,
                key: key.into(),
                targets,
                value,
            });
        }
    }

    /// Leaf key paths written in the TOML file itself, before any override.
    /// Pair with `override_record` to label each key `default | toml | env | cli`.
    fn toml_leaf_paths(content: &str) -> Vec<String> {
        let mut out = Vec::new();
        if let Ok(root) = toml::from_str::<Value>(content) {
            leaves(&root, &mut Vec::new(), &mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = r#"
        [project]
        project_name = "t"
        data_directory = "."

        [[ingress_profile]]
        profile = "videomap"

        [[ingress_profile.tasks]]
        video_dir = "N"
        task_type = "object_recognition"
        meters_per_pixel = { method = "arm_length", meters = 0.35 }
        [ingress_profile.tasks.task_params.exploration]
        max_gap_seconds = 0.5

        [[ingress_profile.tasks]]
        video_dir = "Y"
        task_type = "y_maze"
        meters_per_pixel = { method = "arm_length", meters = 0.35 }
        [[ingress_profile.tasks.variants]]
        name = "a"
        [[ingress_profile.tasks.variants]]
        name = "b"
    "#;

    fn parse(cli: &[&str], env: &str) -> Result<IngressConfig> {
        let cli: Vec<String> = cli.iter().map(|s| s.to_string()).collect();
        let env = env.to_string();
        IngressConfig::parse_with_set_overrides(BASE, &cli, &|k| {
            (k == "BEKIPER_SET").then(|| env.clone())
        })
    }

    fn gap(c: &IngressConfig) -> Option<f64> {
        c.ingress_profile[0].tasks[0]
            .task_params
            .as_ref()?
            .exploration
            .as_ref()?
            .max_gap_seconds
    }

    #[test]
    fn dedicated_shorthands_are_recorded_as_their_generic_keys() {
        let path =
            std::env::temp_dir().join(format!("bekiper_shorthand_{}.toml", std::process::id()));
        std::fs::write(&path, BASE).unwrap();
        let config = IngressConfig::from_file_with_set_overrides_from(
            &path,
            &Default::default(),
            Default::default(),
            Default::default(),
            &[],
            |k| (k == "BEKIPER_NORT_WHISKER_APEX_FRACTION").then(|| "0.85".to_string()),
        )
        .unwrap();
        std::fs::remove_file(&path).ok();
        let record = config
            .override_record
            .iter()
            .find(|r| r.key == "nort.exploration.whisker.apex_fraction")
            .expect("whisker shorthand recorded");
        assert_eq!(record.layer, "env");
        assert_eq!(record.value, Value::Float(0.85));
        assert_eq!(
            record.targets,
            ["ingress_profile[0].tasks[0].task_params.exploration.whisker.apex_fraction"]
        );
    }

    fn from_toml(content: &str, cli: &[&str], env: &[(&str, &str)]) -> IngressConfig {
        let path = std::env::temp_dir().join(format!(
            "bekiper_set_{}_{}.toml",
            std::process::id(),
            content.len() + cli.len()
        ));
        std::fs::write(&path, content).unwrap();
        let cli: Vec<String> = cli.iter().map(|s| s.to_string()).collect();
        let config = IngressConfig::from_file_with_set_overrides_from(
            &path,
            &Default::default(),
            Default::default(),
            Default::default(),
            &cli,
            |k| {
                env.iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| v.to_string())
            },
        )
        .unwrap();
        std::fs::remove_file(&path).ok();
        config
    }

    fn layer_of(c: &IngressConfig, key: &str) -> &'static str {
        c.override_record
            .iter()
            .rev()
            .find(|r| r.key == key)
            .unwrap()
            .layer
    }

    #[test]
    fn cli_set_beats_dedicated_env_overlays() {
        let key = "aggregate.output_filename";
        let c = from_toml(
            BASE,
            &[&format!("{key}=cli.ods")],
            &[("BEKIPER_AGGREGATE_OUTPUT", "env.ods")],
        );
        assert_eq!(c.aggregate.output_filename, "cli.ods");
        assert_eq!(layer_of(&c, key), "cli");
        // env alone still beats the TOML default.
        let c = from_toml(BASE, &[], &[("BEKIPER_AGGREGATE_OUTPUT", "env.ods")]);
        assert_eq!(c.aggregate.output_filename, "env.ods");
        // Dwell env vs `--set` on the same key; the dwell shorthand records as `env`.
        let dwell = "ymaze.ymaze.temporal_policy.min_duration_seconds";
        let envs = [("BEKIPER_YMAZE_DWELL_SECONDS", "2.0")];
        let dwell_of = |c: &IngressConfig| {
            c.ingress_profile[0].tasks[1]
                .task_params
                .as_ref()
                .unwrap()
                .ymaze
                .as_ref()
                .unwrap()
                .temporal_policy
                .as_ref()
                .unwrap()
                .min_duration_seconds
        };
        let c = from_toml(BASE, &[&format!("{dwell}=0.5")], &envs);
        assert_eq!(dwell_of(&c), 0.5);
        assert_eq!(layer_of(&c, dwell), "cli");
        assert_eq!(dwell_of(&from_toml(BASE, &[], &envs)), 2.0);
    }

    #[test]
    fn bare_key_also_writes_variants_that_set_it() {
        let toml = BASE.replace(
            "name = \"b\"",
            "name = \"b\"\n        [ingress_profile.tasks.variants.ymaze]\n        proximal_threshold = 0.9",
        );
        let c = from_toml(&toml, &["ymaze.ymaze.proximal_threshold=0.3"], &[]);
        let tasks = &c.ingress_profile[0].tasks;
        let prox = |p: &super::super::task_overrides::TaskParamsOverrides| {
            p.ymaze.as_ref().and_then(|y| y.proximal_threshold)
        };
        assert_eq!(prox(tasks[1].task_params.as_ref().unwrap()), Some(0.3));
        assert_eq!(prox(&tasks[1].variants[0].task_params), None);
        assert_eq!(prox(&tasks[1].variants[1].task_params), Some(0.3));
    }

    #[test]
    fn typed_roundtrip_is_lossless_for_imen() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/neo/imen/bekiper_config.toml");
        let content = std::fs::read_to_string(path).unwrap();
        let a = from_toml(&content, &[], &[]);
        let mut b = a.clone();
        b.reapply_cli_sets(&[]).unwrap();
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::to_value(&b).unwrap()
        );
    }

    #[test]
    fn precedence_toml_env_cli() {
        assert_eq!(gap(&parse(&[], "").unwrap()), Some(0.5));
        let env = parse(&[], "nort.exploration.max_gap_seconds=0.7").unwrap();
        assert_eq!(gap(&env), Some(0.7));
        let cli = parse(
            &["object_recognition.exploration.max_gap_seconds=0.9"],
            "nort.exploration.max_gap_seconds=0.7",
        )
        .unwrap();
        assert_eq!(gap(&cli), Some(0.9));
        let layers: Vec<_> = cli.override_record.iter().map(|r| r.layer).collect();
        assert_eq!(layers, ["env", "cli"]);
        assert_eq!(
            cli.override_record[1].targets,
            ["ingress_profile[0].tasks[0].task_params.exploration.max_gap_seconds"]
        );
        assert!(cli.toml_paths.iter().any(|p| p == "ingress_profile[0].tasks[0].task_params.exploration.max_gap_seconds"));
    }

    #[test]
    fn variant_and_root_targets_and_value_parsing() {
        let c = parse(
            &[
                "ymaze@b.ymaze.proximal_threshold=0.3",
                "post_analysis.ymaze_primary_thresholds=[0,25]",
                "aggregate.output_filename=x.ods",
            ],
            "nort.exploration.min_bout_seconds=2",
        )
        .unwrap();
        let tasks = &c.ingress_profile[0].tasks;
        let prox = |i: usize| {
            tasks[1].variants[i]
                .task_params
                .ymaze
                .as_ref()
                .and_then(|y| y.proximal_threshold)
        };
        assert_eq!((prox(0), prox(1)), (None, Some(0.3)));
        assert_eq!(
            tasks[0]
                .task_params
                .as_ref()
                .unwrap()
                .exploration
                .as_ref()
                .unwrap()
                .min_bout_seconds,
            Some(2.0)
        );
        assert_eq!(c.aggregate.output_filename, "x.ods");
        assert_eq!(
            parse_pair("k=[0, 25]").unwrap().1,
            Value::Array(vec![0.into(), 25.into()])
        );
        assert_eq!(parse_pair("k=true").unwrap().1, Value::Boolean(true));
        assert_eq!(parse_pair("k=a b").unwrap().1, Value::String("a b".into()));
    }

    #[test]
    fn raw_table_helpers_apply_root_paths_and_reject_unknown_keys() {
        let mut root: Value = toml::from_str("[review]\nworkers = 2\n").unwrap();
        let env = |k: &str| (k == "BEKIPER_SET").then(|| "review.workers=4".to_string());
        let record = apply_set_overrides(
            &mut root,
            &["review.presentation.nort.title=\"NOR\"".into()],
            &env,
        )
        .unwrap();
        assert_eq!(root["review"]["workers"].as_integer(), Some(4));
        assert_eq!(
            root["review"]["presentation"]["nort"]["title"].as_str(),
            Some("NOR")
        );
        assert_eq!(record[0].layer, "env");
        assert_eq!(record[1].targets, ["review.presentation.nort.title"]);
        let typed = serde_json::json!({"review": {"workers": 4, "presentation": {"nort": {"title": "NOR"}}}});
        ensure_overrides_retained(&typed, &record).unwrap();
        let bogus = apply_set_overrides(&mut root, &["review.bogus=1".into()], &|_| None).unwrap();
        assert!(ensure_overrides_retained(&typed, &bogus).is_err());
        assert_eq!(
            render(&parse_target("ingress_profile[0].tasks[2].task_params.x")),
            "ingress_profile[0].tasks[2].task_params.x"
        );
    }

    #[test]
    fn bad_keys_fail_loudly() {
        let err = |cli: &[&str]| parse(cli, "").unwrap_err().to_string();
        assert!(
            err(&["nort.exploration.no_such_knob=1"]).contains("nort.exploration.no_such_knob")
        );
        assert!(err(&["no_such_root.x=1"]).contains("no_such_root.x"));
        assert!(err(&["oft.min_dwell_seconds=1"]).contains("no `oft` task"));
        assert!(err(&["ymaze@zzz.ymaze.proximal_threshold=1"]).contains("variant named `zzz`"));
        assert!(err(&["novalue"]).contains("KEY=VALUE"));
        assert!(err(&["nosuch@v.x=1"]).contains("not a task type"));
        assert!(err(&["nort=1"]).contains("no parameter path"));
        assert!(err(&["project.project_name.x=1"]).contains("is not a table"));
    }

    #[test]
    fn profile_level_tasks_are_targeted_and_retained_is_structural() {
        let mut root: Value =
            toml::from_str("[[ingress_profile]]\ntask_type = \"object_recognition\"\n").unwrap();
        let record = apply_set_overrides(
            &mut root,
            &["nort.exploration.max_gap_seconds=1".into()],
            &|_| None,
        )
        .unwrap();
        assert_eq!(
            record[0].targets,
            ["ingress_profile[0].task_params.exploration.max_gap_seconds"]
        );
        let table = |s: &str| toml::from_str::<Value>(s).unwrap();
        assert!(retained(
            &table("a = { b = 1, c = 2 }"),
            &table("a = { b = 1 }")
        ));
        assert!(!retained(&table("a = { b = 1 }"), &table("a = { b = 2 }")));
        assert_eq!(
            from_json(&serde_json::Value::Null),
            Value::Table(Default::default())
        );
    }

    #[test]
    fn whisker_length_shorthand_is_recorded_as_a_meters_table() {
        let path = std::env::temp_dir().join(format!("bekiper_wlen_{}.toml", std::process::id()));
        std::fs::write(&path, BASE).unwrap();
        let config = IngressConfig::from_file_with_set_overrides_from(
            &path,
            &Default::default(),
            Default::default(),
            Default::default(),
            &[],
            |k| (k == "BEKIPER_NORT_WHISKER_LENGTH_M").then(|| "0.02".to_string()),
        )
        .unwrap();
        std::fs::remove_file(&path).ok();
        assert!(
            config
                .override_record
                .iter()
                .any(|r| r.key == "nort.exploration.whisker.length" && r.layer == "env")
        );
    }
}
