from .test_inspection_dir_with_features import *


class TestPlotFeatureInspections:
    @staticmethod
    def _saved_stems(result) -> set[str]:
        stems: set[str] = set()
        for line in result.output.splitlines():
            if line.startswith("Saved: "):
                stems.add(Path(line.removeprefix("Saved: ").strip()).stem)
        return stems

    def test_plot_emits_feature_files(self, inspection_dir_with_features: Path):
        output_dir = inspection_dir_with_features / "figures"
        result = runner.invoke(
            app,
            [
                "inspect",
                "plot",
                str(inspection_dir_with_features),
                "-o",
                str(output_dir),
                "-f",
                ".png",
            ],
        )
        assert result.exit_code == 0, result.output
        stems = self._saved_stems(result)
        assert "story_object_1" in stems
        assert "story_object_2" in stems
        assert "story_object_decomposition_olfaction_object_1" in stems
        assert "story_object_decomposition_olfaction_object_2" in stems
        assert "story_object_decomposition_body_proximity" in stems
        assert "feature_trajectory_trajectory_nose" in stems
        assert "story_objects" in stems
        assert "story_object_decomposition_olfaction" in stems
        assert not any(
            stem.startswith("feature_heuristic_decomposition_heuristic_decomposition_body_proximity_object_")
            for stem in stems
        )

    def test_only_targets_decomposition_feature(self, inspection_dir_with_features: Path):
        output_dir = inspection_dir_with_features / "figures_only_decomp"
        result = runner.invoke(
            app,
            [
                "inspect",
                "plot",
                str(inspection_dir_with_features),
                "-o",
                str(output_dir),
                "-f",
                ".png",
                "--only",
                "feature_heuristic_decomposition*",
            ],
        )
        assert result.exit_code == 0, result.output
        assert self._saved_stems(result) == {
            "story_object_decomposition_body_proximity",
            "story_object_decomposition_olfaction",
            "story_object_decomposition_olfaction_object_1",
            "story_object_decomposition_olfaction_object_2",
        }

    def test_skip_excludes_decomposition_feature(self, inspection_dir_with_features: Path):
        output_dir = inspection_dir_with_features / "figures_skip_decomp"
        result = runner.invoke(
            app,
            [
                "inspect",
                "plot",
                str(inspection_dir_with_features),
                "-o",
                str(output_dir),
                "-f",
                ".png",
                "--skip",
                "feature_heuristic_decomposition*",
            ],
        )
        assert result.exit_code == 0, result.output
        stems = self._saved_stems(result)
        assert "story_object_decomposition_body_proximity" not in stems
        assert "story_object_decomposition_olfaction_object_1" not in stems
        assert "story_object_decomposition_olfaction_object_2" not in stems
        assert "story_object_1" in stems
        assert "story_object_2" in stems
        assert "feature_trajectory_trajectory_nose" in stems

    def test_no_stories_keeps_individual_decomposition_features(self, inspection_dir_with_features: Path):
        output_dir = inspection_dir_with_features / "figures_no_stories"
        result = runner.invoke(
            app,
            [
                "inspect",
                "plot",
                str(inspection_dir_with_features),
                "-o",
                str(output_dir),
                "-f",
                ".png",
                "--no-stories",
            ],
        )
        assert result.exit_code == 0, result.output
        stems = self._saved_stems(result)
        assert "feature_heuristic_heuristic_olfaction_object_1" in stems
        assert "feature_heuristic_heuristic_olfaction_object_2" in stems
        assert "feature_heuristic_decomposition_heuristic_decomposition_body_proximity_object_1" in stems
        assert "feature_heuristic_decomposition_heuristic_decomposition_body_proximity_object_2" in stems
        assert "feature_heuristic_decomposition_heuristic_decomposition_olfaction_object_1" in stems
        assert "feature_heuristic_decomposition_heuristic_decomposition_olfaction_object_2" in stems

    def test_plot_refreshes_merged_stories_and_preserves_other_formats(self, inspection_dir_with_features: Path):
        output_dir = inspection_dir_with_features / "figures_cleanup"
        output_dir.mkdir(parents=True, exist_ok=True)
        stale_objects = output_dir / "story_objects.png"
        stale_olfaction_png = output_dir / "story_object_decomposition_olfaction.png"
        stale_olfaction_webp = output_dir / "story_object_decomposition_olfaction.webp"
        stale_objects.write_bytes(b"old")
        stale_olfaction_png.write_bytes(b"old")
        stale_olfaction_webp.write_bytes(b"old")

        result = runner.invoke(
            app,
            [
                "inspect",
                "plot",
                str(inspection_dir_with_features),
                "-o",
                str(output_dir),
                "-f",
                ".png",
            ],
        )

        assert result.exit_code == 0, result.output
        assert stale_objects.read_bytes().startswith(b"\x89PNG")
        assert stale_olfaction_png.read_bytes().startswith(b"\x89PNG")
        assert stale_olfaction_webp.read_bytes() == b"old"
        assert "Removed stale:" not in result.output

    def test_filtering_a_single_object_preserves_existing_composites(self, inspection_dir_with_features: Path):
        output_dir = inspection_dir_with_features / "filtered"
        output_dir.mkdir()
        merged = output_dir / "story_objects.png"
        merged.write_bytes(b"existing composite")
        result = runner.invoke(
            app,
            [
                "inspect",
                "plot",
                str(inspection_dir_with_features),
                "-o",
                str(output_dir),
                "-f",
                ".png",
                "--only",
                "story_object_1.png",
            ],
        )
        assert result.exit_code == 0, result.output
        assert merged.read_bytes() == b"existing composite"

    def test_plot_skips_features_without_video(self, inspection_dir: Path):
        # The default fixture has no `video_path`; the per-feature loop
        # should silently skip without breaking the rest of the plots.
        output_dir = inspection_dir / "figures"
        result = runner.invoke(
            app,
            [
                "inspect",
                "plot",
                str(inspection_dir),
                "-o",
                str(output_dir),
                "-f",
                ".png",
            ],
        )
        assert result.exit_code == 0
        # No feature_*.png files (manifest has no feature_inspections),
        # but the existing summary/timeline/coords figures must still be
        # there.
        non_feature = [p for p in output_dir.glob("*.png") if not p.name.startswith("feature_")]
        assert len(non_feature) >= 1


class TestPlotFilters:
    """Tests for --only, --skip, --features-only, --no-features."""

    def _run_plot(self, inspection_dir: Path, *extra_args: str) -> tuple[object, Path]:
        output_dir = inspection_dir / "figures_filter"
        result = runner.invoke(
            app,
            [
                "inspect",
                "plot",
                str(inspection_dir),
                "-o",
                str(output_dir),
                "-f",
                ".png",
                "--no-stories",
                *extra_args,
            ],
        )
        return result, output_dir

    def _saved_stems(self, result) -> set[str]:
        """Extract figure stems from 'Saved: ...' lines in CLI output."""
        stems: set[str] = set()
        for line in result.output.splitlines():
            if line.startswith("Saved: "):
                p = Path(line.removeprefix("Saved: ").strip())
                stems.add(p.stem)
        return stems

    def test_no_filter_emits_all(self, inspection_dir: Path):
        result, _ = self._run_plot(inspection_dir)
        assert result.exit_code == 0
        stems = self._saved_stems(result)
        assert "summary" in stems
        assert any(s.startswith("timeline_") for s in stems)

    def test_only_summary(self, inspection_dir: Path):
        result, _ = self._run_plot(inspection_dir, "--only", "summary*")
        assert result.exit_code == 0
        stems = self._saved_stems(result)
        assert "summary" in stems
        assert all(s == "summary" for s in stems)

    def test_only_timeline_glob(self, inspection_dir: Path):
        result, _ = self._run_plot(inspection_dir, "--only", "timeline_*")
        assert result.exit_code == 0
        stems = self._saved_stems(result)
        assert len(stems) >= 1
        assert all(s.startswith("timeline_") for s in stems)

    def test_skip_timeline(self, inspection_dir: Path):
        result, _ = self._run_plot(inspection_dir, "--skip", "timeline_*")
        assert result.exit_code == 0
        stems = self._saved_stems(result)
        assert not any(s.startswith("timeline_") for s in stems)
        assert "summary" in stems

    def test_skip_coords(self, inspection_dir: Path):
        result, _ = self._run_plot(inspection_dir, "--skip", "coords_*")
        assert result.exit_code == 0
        stems = self._saved_stems(result)
        assert not any(s.startswith("coords_") for s in stems)
        assert "summary" in stems

    def test_only_and_skip_combined(self, inspection_dir: Path):
        result, _ = self._run_plot(inspection_dir, "--only", "timeline_*", "--skip", "timeline_olfaction*")
        assert result.exit_code == 0
        stems = self._saved_stems(result)
        assert all(s.startswith("timeline_") for s in stems)
        assert "timeline_olfaction" not in stems

    def test_features_only_skips_heuristics(self, inspection_dir: Path):
        result, _ = self._run_plot(inspection_dir, "--features-only")
        assert result.exit_code == 0
        stems = self._saved_stems(result)
        assert len(stems) == 0

    def test_no_features_keeps_heuristics(self, inspection_dir: Path):
        result, _ = self._run_plot(inspection_dir, "--no-features")
        assert result.exit_code == 0
        stems = self._saved_stems(result)
        assert "summary" in stems
        assert any(s.startswith("timeline_") for s in stems)

    def test_features_only_and_no_features_errors(self, inspection_dir: Path):
        result, _ = self._run_plot(inspection_dir, "--features-only", "--no-features")
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower() or "mutually exclusive" in (result.stdout or "")


class TestVideoCommand:
    def test_video_wrapper_forwards_feature_args(self, inspection_dir: Path, monkeypatch: pytest.MonkeyPatch):
        calls: dict[str, object] = {}

        def fake_run(args, env=None, check=False):
            calls["args"] = args
            calls["env"] = env
            calls["check"] = check
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr(inspect_cli.subprocess, "run", fake_run)

        output_path = inspection_dir / "trajectory.mp4"
        result = runner.invoke(
            app,
            [
                "inspect",
                "video",
                str(inspection_dir),
                "--feature",
                "trajectory_nose",
                "--codec",
                "mpeg4",
                "--ffmpeg-preset",
                "fast",
                "-o",
                str(output_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert calls["args"] == [
            "bekiper",
            "inspect",
            "video",
            str(inspection_dir.resolve()),
            "--codec",
            "mpeg4",
            "-o",
            str(output_path),
            "--feature",
            "trajectory_nose",
            "--ffmpeg-preset",
            "fast",
        ]
        assert calls["check"] is False
        env = calls["env"]
        assert isinstance(env, dict)
        assert env["BEKIPER_PYTHON_VIDEO_SHIM"] == "1"

    def test_video_wrapper_forwards_batch_args(self, inspection_dir: Path, monkeypatch: pytest.MonkeyPatch):
        calls: dict[str, object] = {}

        def fake_run(args, env=None, check=False):
            calls["args"] = args
            calls["env"] = env
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr(inspect_cli.subprocess, "run", fake_run)

        output_dir = inspection_dir / "videos"
        config = inspection_dir / "bekiper_config.toml"
        result = runner.invoke(
            app,
            [
                "inspect",
                "video",
                str(inspection_dir),
                "--all-features",
                "--output-dir",
                str(output_dir),
                "--config",
                str(config),
            ],
        )

        assert result.exit_code == 0, result.output
        assert calls["args"] == [
            "bekiper",
            "inspect",
            "video",
            str(inspection_dir.resolve()),
            "--codec",
            "h264",
            "--output-dir",
            str(output_dir),
            "--all-features",
            "--config",
            str(config),
        ]

    def test_video_wrapper_forwards_scored_args(self, inspection_dir: Path, monkeypatch: pytest.MonkeyPatch):
        calls: dict[str, object] = {}

        def fake_run(args, env=None, check=False):
            calls["args"] = args
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr(inspect_cli.subprocess, "run", fake_run)

        source = inspection_dir / "rebased.mp4"
        result = runner.invoke(
            app,
            ["inspect", "video", str(inspection_dir), "--scored", "--source-video", str(source), "--crf", "20"],
        )

        assert result.exit_code == 0, result.output
        assert calls["args"][-5:] == ["--source-video", str(source), "--crf", "20", "--scored"]

    def test_video_wrapper_propagates_exit_code(self, inspection_dir: Path, monkeypatch: pytest.MonkeyPatch):
        def fake_run(args, env=None, check=False):
            return subprocess.CompletedProcess(args, 7)

        monkeypatch.setattr(inspect_cli.subprocess, "run", fake_run)

        result = runner.invoke(app, ["inspect", "video", str(inspection_dir)])
        assert result.exit_code == 7

    def test_video_wrapper_reentry_guard(self, inspection_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BEKIPER_PYTHON_VIDEO_SHIM", "1")

        result = runner.invoke(app, ["inspect", "video", str(inspection_dir)])
        assert result.exit_code != 0
        assert isinstance(result.exception, RuntimeError)
        assert "re-entered itself" in str(result.exception)


class TestDetectCommand:
    def test_detect_list(self):
        result = runner.invoke(app, ["detect", "list"])
        assert result.exit_code == 0
        assert "nort" in result.output
        assert "open_field" in result.output
