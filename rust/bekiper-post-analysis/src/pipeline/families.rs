//! Content-addressed figure families shared by the contexts of one run.
//!
//! A figure page is fully determined by its plotting-bundle record, so its
//! rendered files are stored once under `families/<assay>/<variant>/<digest>/`
//! and hardlinked (copied across filesystems) into every context that draws
//! the same page. OFT/NORT pages do not depend on the Y-maze threshold and are
//! therefore rendered by one context only.
use super::*;
use sha2::{Digest, Sha256};

/// Config key carrying the store path to a context; removed before any
/// manifest records the configuration.
pub(super) const KEY_STORE: &str = "_family_store";

pub(super) fn take(config: &mut Record) -> Option<PathBuf> {
    config
        .as_object_mut()?
        .remove(KEY_STORE)?
        .as_str()
        .map(PathBuf::from)
}

fn slug(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// `(assay, variant)` of a page: the assay named by the page, the variant its
/// first panel records (`default` when the page spans none).
fn family_of_page(figure: &Record) -> (String, String) {
    let name = figure["name"].as_str().unwrap_or_default();
    let assay = ["oft", "ymaze", "nort"]
        .into_iter()
        .find(|assay| name.contains(assay))
        .unwrap_or("shared");
    let variant = figure["panels"][0]["session_partition"]["variant"]
        .as_str()
        .unwrap_or("default");
    (slug(assay), slug(variant))
}

fn digest(figure: &Record, color_mode: &str) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(serde_json::to_vec(figure)?);
    hash.update(color_mode.as_bytes());
    Ok(hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn files(figure: &Record) -> Result<Vec<String>> {
    let name = figure["name"].as_str().context("figure needs name")?;
    figure["formats"]
        .as_array()
        .context("figure needs formats")?
        .iter()
        .map(|format| Ok(format!("{name}.{}", format.as_str().context("format")?)))
        .collect()
}

fn link_or_copy(from: &Path, to: &Path) -> Result<()> {
    if std::fs::hard_link(from, to).is_err() {
        std::fs::copy(from, to)?;
    }
    Ok(())
}

/// Stored page directory when every file of `figure` is already present.
pub(super) struct Lookup {
    pub hits: Vec<usize>,
    pub misses: Vec<usize>,
    keys: Vec<PathBuf>,
}

pub(super) fn restore(store: &Path, directory: &Path, figures: &[Record]) -> Result<Lookup> {
    let color_mode = std::env::var("BEKIPER_PLOT_COLOR_MODE").unwrap_or_default();
    let mut lookup = Lookup {
        hits: vec![],
        misses: vec![],
        keys: vec![],
    };
    for (index, figure) in figures.iter().enumerate() {
        let (assay, variant) = family_of_page(figure);
        let key = store
            .join(assay)
            .join(variant)
            .join(digest(figure, &color_mode)?);
        lookup.keys.push(key.clone());
        if key.join("complete").is_file() {
            for file in files(figure)? {
                link_or_copy(&key.join(&file), &directory.join(&file))?;
            }
            lookup.hits.push(index);
        } else {
            lookup.misses.push(index);
        }
    }
    Ok(lookup)
}

/// Publish freshly rendered pages. A concurrent context that stored the same
/// page first wins; the files are byte-identical by construction.
pub(super) fn publish(
    lookup: &Lookup,
    directory: &Path,
    figures: &[Record],
    renderer: &Record,
    store: &Path,
) -> Result<()> {
    for &index in &lookup.misses {
        let key = &lookup.keys[index];
        let parent = key.parent().context("family key parent")?;
        std::fs::create_dir_all(parent)?;
        let staging = tempfile::Builder::new()
            .prefix(".page-")
            .tempdir_in(parent)?;
        for file in files(&figures[index])? {
            link_or_copy(&directory.join(&file), &staging.path().join(&file))?;
        }
        std::fs::write(staging.path().join("complete"), b"")?;
        // Already present: a concurrent context stored the same page first.
        let _ = std::fs::rename(staging.path(), key);
    }
    let record = store.join("renderer.json");
    if !record.exists() && renderer.is_object() {
        let _ = std::fs::write(&record, serde_json::to_vec(renderer)?);
    }
    Ok(())
}

pub(super) fn renderer_record(store: &Path) -> Record {
    std::fs::read(store.join("renderer.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stored_page_is_restored_byte_identically_and_only_misses_render() {
        let root = tempfile::tempdir().unwrap();
        let store = root.path().join("families");
        let (first, second) = (root.path().join("a"), root.path().join("b"));
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        let page = |name: &str| json!({"name":name,"formats":["png"],"panels":[{}]});
        let figures = [page("figure_oft_page"), page("figure_ymaze_page")];
        let cold = restore(&store, &first, &figures).unwrap();
        assert_eq!((cold.hits.len(), cold.misses.len()), (0, 2));
        for figure in &figures {
            let name = figure["name"].as_str().unwrap();
            std::fs::write(first.join(format!("{name}.png")), name).unwrap();
        }
        publish(&cold, &first, &figures, &Value::Null, &store).unwrap();
        // A context with a different Y-maze page reuses only the OFT page.
        let varied = [page("figure_oft_page"), page("figure_ymaze_other")];
        let warm = restore(&store, &second, &varied).unwrap();
        assert_eq!((warm.hits.clone(), warm.misses.clone()), (vec![0], vec![1]));
        assert_eq!(
            std::fs::read(second.join("figure_oft_page.png")).unwrap(),
            b"figure_oft_page"
        );
        assert!(!second.join("figure_ymaze_other.png").exists());
    }

    #[test]
    fn helpers_slug_overwrite_and_read_back_the_renderer_record() {
        assert_eq!(slug("a b/c-d"), "a_b_c-d");
        let root = tempfile::tempdir().unwrap();
        let (from, to) = (root.path().join("from"), root.path().join("to"));
        std::fs::write(&from, "new").unwrap();
        std::fs::write(&to, "old").unwrap();
        // The link fails on an existing target; the copy fallback replaces it.
        link_or_copy(&from, &to).unwrap();
        assert_eq!(std::fs::read(&to).unwrap(), b"new");

        let store = root.path().join("store");
        std::fs::create_dir_all(&store).unwrap();
        assert_eq!(renderer_record(&store), Value::Null);
        std::fs::write(store.join("renderer.json"), "not json").unwrap();
        assert_eq!(renderer_record(&store), Value::Null);
        std::fs::remove_file(store.join("renderer.json")).unwrap();
        let lookup = Lookup {
            hits: vec![],
            misses: vec![],
            keys: vec![],
        };
        publish(&lookup, root.path(), &[], &json!({"dpi": 150}), &store).unwrap();
        assert_eq!(renderer_record(&store), json!({"dpi": 150}));

        let mut config = json!({KEY_STORE: store.to_str().unwrap(), "other": 1});
        assert_eq!(take(&mut config), Some(store));
        assert!(config.get(KEY_STORE).is_none());
        assert_eq!(take(&mut config), None);
    }
}
