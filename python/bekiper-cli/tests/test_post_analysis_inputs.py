import json
import zipfile

import pytest

from reference_post_analysis.inputs import normalize, read_table, resolved_config


def test_ods_adapter_preserves_nulls_booleans_sequence_and_json_provenance(tmp_path):
    evidence = {"geometry_version": "imen_radial_vertex_span_v1", "apply_dwell_filter": False}
    content = """<office:document-content
      xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
      xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
      xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
      <office:body><office:spreadsheet><table:table table:name="ymaze_proximal45">
        <table:table-row>
          <table:table-cell><text:p>animal</text:p></table:table-cell>
          <table:table-cell><text:p>valid_arm_entry_sequence</text:p></table:table-cell>
          <table:table-cell><text:p>excluded</text:p></table:table-cell>
          <table:table-cell><text:p>scoring_provenance</text:p></table:table-cell>
          <table:table-cell><text:p>freezing_s</text:p></table:table-cell>
        </table:table-row><table:table-row>
          <table:table-cell office:value-type="float" office:value="99"/>
          <table:table-cell><text:p>A A B</text:p></table:table-cell>
          <table:table-cell office:value-type="boolean" office:boolean-value="false"/>
          <table:table-cell><text:p>PROVENANCE</text:p></table:table-cell>
          <table:table-cell/>
        </table:table-row>
      </table:table></office:spreadsheet></office:body></office:document-content>"""
    path = tmp_path / "aggregate.ods"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("content.xml", content.replace("PROVENANCE", json.dumps(evidence)))
    row = normalize(read_table(path, "ymaze_proximal45")[0], {})
    assert row["animal_id"] == "99"
    assert row["excluded"] is False
    assert row["freezing_s"] is None
    assert row["valid_arm_entry_sequence"] == ["A", "A", "B"]
    assert row["scoring_provenance"] == evidence
    with pytest.raises(ValueError, match="explicit sheet"):
        read_table(path)


def test_output_precedence_without_process_environment_mutation(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"run_mode": "development", "outputs": {"directory": "config_output"}}))
    assert resolved_config(path, env={})["outputs"]["directory"] == "config_output"
    env = {"BEKIPER_POST_ANALYSIS_OUTPUT": "env_output"}
    assert resolved_config(path, env=env)["outputs"]["directory"] == "env_output"
    assert resolved_config(path, output=tmp_path / "cli", env=env)["outputs"]["directory"] == str(tmp_path / "cli")


def test_requested_output_precedence_and_validation(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"run_mode": "development", "outputs": {"requested_outputs": ["metrics"]}}))
    assert resolved_config(path, env={})["outputs"]["requested_outputs"] == ["metrics"]
    env = {"BEKIPER_POST_ANALYSIS_REQUESTED_OUTPUTS": "models, within_group_tests"}
    assert resolved_config(path, env=env)["outputs"]["requested_outputs"] == ["models", "within_group_tests"]
    assert resolved_config(path, requested_outputs=["figures"], env=env)["outputs"]["requested_outputs"] == ["figures"]
    with pytest.raises(ValueError, match="requested_outputs"):
        resolved_config(path, requested_outputs=["typo"], env={})
    with pytest.raises(ValueError, match="requested_outputs"):
        resolved_config(path, requested_outputs=[], env={})


def test_csv_duplicate_headers_and_conflicting_aliases_are_rejected(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("animal,animal\na,a\n")
    with pytest.raises(ValueError, match="duplicate CSV"):
        read_table(path)
    with pytest.raises(ValueError, match="alias collision"):
        normalize({"animal": "a", "animal_id": "b"}, {})


def test_sensitivity_threshold_precedence_and_validation(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {"run_mode": "development", "assays": {"ymaze": {"requested_sensitivity_thresholds_arm_length_pct": [25]}}}
        )
    )
    key = "requested_sensitivity_thresholds_arm_length_pct"
    env = {"BEKIPER_POST_ANALYSIS_SENSITIVITY_THRESHOLDS": "45,65"}
    assert resolved_config(path, env={})["assays"]["ymaze"][key] == [25]
    assert resolved_config(path, env=env)["assays"]["ymaze"][key] == [45, 65]
    assert resolved_config(path, sensitivity_thresholds=[65], env=env)["assays"]["ymaze"][key] == [65]
    assert resolved_config(path, sensitivity_thresholds=[], env=env)["assays"]["ymaze"][key] == []
    with pytest.raises(ValueError, match="sensitivity"):
        resolved_config(path, sensitivity_thresholds=[25, 25], env={})
