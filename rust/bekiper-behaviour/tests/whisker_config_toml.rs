//! TOML shape of `[...exploration.whisker]` (serde types live in bekiper-feature).
use bekiper_feature::heuristic::solo::whisker_config::{WhiskerLengthSpec, WhiskerOverrides};

#[test]
fn meters_and_age_table_shapes_parse() {
    let o: WhiskerOverrides = toml::from_str("enabled = true\nlength = { meters = 0.02 }").unwrap();
    assert_eq!(o.params().unwrap().unwrap().length_m, 0.02);
    let t: WhiskerOverrides = toml::from_str(
        "enabled = true\n[length]\nmethod = \"age_table\"\nsource = \"mice_overview.xlsx\"\n\
         table = [[0, 0.0117], [400, 0.0117]]",
    )
    .unwrap();
    let Some(WhiskerLengthSpec::AgeTable(spec)) = &t.length else {
        panic!("age table expected")
    };
    assert_eq!(spec.animal_column, "Animal ID");
    assert_eq!(spec.post_age_column, "End age (days)");
    spec.validate().unwrap();
    assert!(
        t.params().is_err(),
        "unresolved age table must not fall back"
    );
    assert!(toml::from_str::<WhiskerOverrides>("enabled = true\nbogus = 1").is_err());
}
