# bekiper

Behavioural kinematics analysis for rodent experiments. bekiper turns
markerless pose estimates (DeepLabCut, SLEAP, LightningPose) and arena geometry
into per-trial behavioural metrics and registered cohort statistics for:

| Task | Examples of metrics |
|---|---|
| Open field | Distance, speed, centre/periphery time and entries |
| Novel object recognition (NORT) | Exploration bouts (nose and whisker contact), discrimination index |
| Y-maze | Arm entries, spontaneous alternation, dwell-filtered arm sequences |

The engine is written in Rust (`rust/`). Figures and inspection plots are
rendered by the Python packages in `python/`.

This repository contains the analysis engine. The annotation application and
multi-user registry are not included (see [COMMERCIAL.md](COMMERCIAL.md)).

## Build

With Nix (provides ROCm, ONNX Runtime and the Python environment):

```sh
nix develop
cargo build --release -p bekiper-cli -p bekiper-post-analysis
```

or `nix build .#bekiper`.

## Reproducing the Neo/Imen analysis

`examples/neo/imen/bekiper_config.toml` is the configuration used for the
Neo/Imen study; `post_analysis_plan.toml` records the approved analysis
decisions. Place the published dataset (videos, DeepLabCut pose files, arena
annotations, animal metadata) in that directory, or point `data_directory` at
it, set `output_dir` to a new, empty directory, then run:

```sh
bekiper run --config examples/neo/imen/bekiper_config.toml
```

The published results were produced with exactly this configuration; only
`data_directory` and `output_dir` differed.

To confirm that your build is the code that produced a published run, see
[PROVENANCE.md](PROVENANCE.md#engine-identity).

## Citing

Cite the exact version you used. Each release has its own DOI on Zenodo, and
[CITATION.cff](CITATION.cff) has the metadata.

## License

AGPL-3.0-only, with commercial licenses available. See [COMMERCIAL.md](COMMERCIAL.md).
