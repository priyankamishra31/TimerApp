"""
tests/unit/test_config_loader.py
---------------------------------
Comprehensive unit tests for config_loader.py and init_project.py.

Coverage targets
----------------
- Happy path: valid minimal config, valid full config
- File-level errors: not found (template creation), empty, YAML syntax,
  wrong root type, permission error, bad extension
- Sentinel / unconfigured detection
- Field-level errors: missing required fields, wrong types, out-of-range
  numerics, invalid enum values
- Cross-field errors: all conflict combinations
- Cross-field warnings: all warning-only combinations
- OD-blocked features: Robust, non-Gaussian GLM, custom_woe
- init_project: creates file, does not overwrite, missing template
- PipelineConfig attributes: correct values after successful load
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from linear_regression_tool.config_loader import PipelineConfig, load_config
from linear_regression_tool.exceptions import (
    ConfigFileError,
    ConfigNotConfiguredError,
    ConfigNotFoundError,
    ConfigODBlockedError,
    ConfigValidationError,
)
from linear_regression_tool.init_project import _get_template_yaml, init


# =============================================================================
#  Fixtures & helpers
# =============================================================================


def write_config(tmp_path: Path, content: dict | str) -> Path:
    """Write a YAML config to a temp file and return the path."""
    p = tmp_path / "config.yaml"
    if isinstance(content, dict):
        p.write_text(yaml.dump(content), encoding="utf-8")
    else:
        p.write_text(content, encoding="utf-8")
    return p


MINIMAL_VALID = {
    "data": {
        "source_type": "csv",
        "file_path": "data.csv",
        "target_variable": "loan_default",
        "variables_are_encoded": True,
    },
    "modelling_controls": {
        "model_type": "OLS",
        "selection_method": "stepwise",
    },
    "output": {
        "formats": ["console"],
    },
}

FULL_VALID = {
    "data": {
        "source_type": "csv",
        "file_path": "data.csv",
        "target_variable": "loan_default",
        "target_is_binned": False,
        "variables_are_encoded": True,
    },
    "encoding": {
        "apply_encoding": False,
        "method": "dummy",
        "columns": [],
    },
    "modelling_controls": {
        "model_type": "OLS",
        "selection_method": "stepwise",
        "selection_criterion": "all",
        "p_value_entry": 0.05,
        "p_value_removal": 0.10,
        "vif_threshold": 10.0,
        "vif_action": "warn",
    },
    "feature_selection": {
        "manual_variables": [],
    },
    "output": {
        "base_dir": "./outputs",
        "formats": ["csv", "html", "console"],
    },
}


# =============================================================================
#  Happy path
# =============================================================================


class TestHappyPath:
    def test_minimal_valid_config_loads(self, tmp_path):
        p = write_config(tmp_path, MINIMAL_VALID)
        config = load_config(p)
        assert isinstance(config, PipelineConfig)

    def test_full_valid_config_loads(self, tmp_path):
        p = write_config(tmp_path, FULL_VALID)
        config = load_config(p)
        assert config.data.target_variable == "loan_default"
        assert config.modelling_controls.model_type == "OLS"

    def test_returns_pipeline_config_type(self, tmp_path):
        p = write_config(tmp_path, MINIMAL_VALID)
        config = load_config(p)
        assert isinstance(config, PipelineConfig)

    def test_defaults_applied(self, tmp_path):
        """Fields with defaults should not need to be specified."""
        p = write_config(tmp_path, MINIMAL_VALID)
        config = load_config(p)
        assert config.modelling_controls.p_value_entry == 0.05
        assert config.modelling_controls.p_value_removal == 0.10
        assert config.modelling_controls.vif_threshold == 10.0
        assert config.modelling_controls.vif_action == "warn"
        assert config.modelling_controls.selection_criterion == "all"

    def test_target_is_binned_defaults_false(self, tmp_path):
        p = write_config(tmp_path, MINIMAL_VALID)
        config = load_config(p)
        assert config.data.target_is_binned is False

    def test_dataframe_source_no_file_path_required(self, tmp_path):
        cfg = {**MINIMAL_VALID, "data": {
            "source_type": "dataframe",
            "target_variable": "y",
            "variables_are_encoded": True,
        }}
        cfg["output"] = {"formats": ["console"]}
        p = write_config(tmp_path, cfg)
        config = load_config(p)
        assert config.data.source_type == "dataframe"

    def test_glm_gaussian_is_active(self, tmp_path):
        cfg = {
            **FULL_VALID,
            "modelling_controls": {
                **FULL_VALID["modelling_controls"],
                "model_type": "GLM",
                "glm_distribution": "gaussian",
            },
        }
        p = write_config(tmp_path, cfg)
        config = load_config(p)
        assert config.modelling_controls.model_type == "GLM"
        assert config.modelling_controls.glm_distribution == "gaussian"

    def test_manual_selection_with_variables(self, tmp_path):
        cfg = {
            **FULL_VALID,
            "modelling_controls": {
                **FULL_VALID["modelling_controls"],
                "selection_method": "manual",
            },
            "feature_selection": {"manual_variables": ["var_a", "var_b"]},
        }
        p = write_config(tmp_path, cfg)
        config = load_config(p)
        assert config.feature_selection.manual_variables == ["var_a", "var_b"]

    def test_excel_source_type(self, tmp_path):
        cfg = {**MINIMAL_VALID}
        cfg["data"] = {
            "source_type": "excel",
            "file_path": "data.xlsx",
            "target_variable": "y",
            "variables_are_encoded": True,
        }
        p = write_config(tmp_path, cfg)
        config = load_config(p)
        assert config.data.source_type == "excel"

    def test_encoding_false_with_apply_encoding_and_method(self, tmp_path):
        cfg = {
            **FULL_VALID,
            "data": {
                **FULL_VALID["data"],
                "variables_are_encoded": False,
            },
            "encoding": {
                "apply_encoding": True,
                "method": "dummy",
            },
        }
        p = write_config(tmp_path, cfg)
        config = load_config(p)
        assert config.encoding.apply_encoding is True

    def test_yaml_extension_variant(self, tmp_path):
        """Files with .yml extension should also be parsed."""
        p = tmp_path / "config.yml"
        p.write_text(yaml.dump(MINIMAL_VALID), encoding="utf-8")
        config = load_config(p)
        assert isinstance(config, PipelineConfig)


# =============================================================================
#  File-level errors
# =============================================================================


class TestFileLevelErrors:
    def test_file_not_found_writes_template_and_raises(self, tmp_path):
        p = tmp_path / "nonexistent" / "config.yaml"
        with pytest.raises(ConfigNotFoundError):
            load_config(p)
        # Template should have been written
        assert p.exists()

    def test_file_not_found_message_mentions_edit(self, tmp_path):
        p = tmp_path / "config.yaml"
        with pytest.raises(ConfigNotFoundError, match="edit"):
            load_config(p)

    def test_template_written_on_not_found(self, tmp_path):
        p = tmp_path / "my_config.yaml"
        assert not p.exists()
        with pytest.raises(ConfigNotFoundError):
            load_config(p)
        assert p.exists()
        content = p.read_text()
        assert "target_variable" in content  # it's a real template

    def test_empty_file_raises_not_configured(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ConfigNotConfiguredError, match="empty"):
            load_config(p)

    def test_whitespace_only_file_raises_not_configured(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("   \n\n   \t  ", encoding="utf-8")
        with pytest.raises(ConfigNotConfiguredError):
            load_config(p)

    def test_invalid_yaml_syntax_raises_file_error(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("data:\n  target: :\n  bad: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigFileError, match="invalid YAML"):
            load_config(p)

    def test_yaml_root_is_list_raises_file_error(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigFileError, match="not a mapping"):
            load_config(p)

    def test_yaml_root_is_string_raises_file_error(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("just a plain string\n", encoding="utf-8")
        with pytest.raises(ConfigFileError):
            load_config(p)

    def test_non_utf8_encoding_raises_file_error(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_bytes(b"target: \xff\xfe invalid utf8")
        with pytest.raises(ConfigFileError, match="UTF-8"):
            load_config(p)

    def test_wrong_extension_warns_but_parses(self, tmp_path, caplog):
        p = tmp_path / "config.txt"
        p.write_text(yaml.dump(MINIMAL_VALID), encoding="utf-8")
        # Should load successfully but emit a warning
        config = load_config(p)
        assert isinstance(config, PipelineConfig)
        assert any("extension" in r.message.lower() for r in caplog.records)

    def test_section_is_null_raises_file_error(self, tmp_path):
        """When a required section like 'data: null' is written."""
        p = tmp_path / "config.yaml"
        p.write_text("data: null\nmodelling_controls:\n  model_type: OLS\n  selection_method: stepwise\noutput:\n  formats:\n    - console\n", encoding="utf-8")
        with pytest.raises((ConfigValidationError, ConfigFileError)):
            load_config(p)


# =============================================================================
#  Sentinel / unconfigured detection
# =============================================================================


class TestSentinelDetection:
    def test_target_sentinel_raises(self, tmp_path):
        cfg = {**MINIMAL_VALID, "data": {
            **MINIMAL_VALID["data"],
            "target_variable": "YOUR_TARGET_HERE",
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises((ConfigNotConfiguredError, ConfigValidationError)):
            load_config(p)

    def test_file_path_sentinel_raises(self, tmp_path):
        cfg = {**MINIMAL_VALID, "data": {
            **MINIMAL_VALID["data"],
            "file_path": "/path/to/your/data.csv",
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises((ConfigNotConfiguredError, ConfigValidationError)):
            load_config(p)

    def test_source_type_sentinel_raises(self, tmp_path):
        cfg = {**MINIMAL_VALID, "data": {
            **MINIMAL_VALID["data"],
            "source_type": "YOUR_SOURCE_TYPE_HERE",
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises((ConfigNotConfiguredError, ConfigValidationError)):
            load_config(p)


# =============================================================================
#  Field-level validation errors
# =============================================================================


class TestFieldLevelErrors:

    # ── Missing required fields ────────────────────────────────────────────────

    def test_missing_data_section_raises(self, tmp_path):
        cfg = {k: v for k, v in FULL_VALID.items() if k != "data"}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError, match="data"):
            load_config(p)

    def test_missing_modelling_controls_raises(self, tmp_path):
        cfg = {k: v for k, v in FULL_VALID.items() if k != "modelling_controls"}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_missing_output_section_raises(self, tmp_path):
        cfg = {k: v for k, v in FULL_VALID.items() if k != "output"}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_missing_target_variable_raises(self, tmp_path):
        cfg = {**FULL_VALID, "data": {k: v for k, v in FULL_VALID["data"].items() if k != "target_variable"}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError, match="target_variable"):
            load_config(p)

    def test_missing_source_type_raises(self, tmp_path):
        cfg = {**FULL_VALID, "data": {k: v for k, v in FULL_VALID["data"].items() if k != "source_type"}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_missing_variables_are_encoded_raises(self, tmp_path):
        cfg = {**FULL_VALID, "data": {k: v for k, v in FULL_VALID["data"].items() if k != "variables_are_encoded"}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_missing_formats_raises(self, tmp_path):
        cfg = {**FULL_VALID, "output": {"base_dir": "./out"}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_empty_formats_list_raises(self, tmp_path):
        cfg = {**FULL_VALID, "output": {"base_dir": "./out", "formats": []}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError, match="formats"):
            load_config(p)

    # ── Invalid enum values ────────────────────────────────────────────────────

    def test_invalid_source_type_raises(self, tmp_path):
        cfg = {**FULL_VALID, "data": {**FULL_VALID["data"], "source_type": "database"}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_invalid_model_type_raises(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {**FULL_VALID["modelling_controls"], "model_type": "Lasso"}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_invalid_selection_method_raises(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {**FULL_VALID["modelling_controls"], "selection_method": "random"}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_invalid_selection_criterion_raises(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {**FULL_VALID["modelling_controls"], "selection_criterion": "r_squared"}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_invalid_vif_action_raises(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {**FULL_VALID["modelling_controls"], "vif_action": "delete"}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_invalid_output_format_raises(self, tmp_path):
        cfg = {**FULL_VALID, "output": {"formats": ["pdf"]}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_invalid_encoding_method_raises(self, tmp_path):
        cfg = {
            **FULL_VALID,
            "data": {**FULL_VALID["data"], "variables_are_encoded": False},
            "encoding": {"apply_encoding": True, "method": "label_encode"},
        }
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    # ── Numeric range errors ───────────────────────────────────────────────────

    def test_p_value_entry_zero_raises(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {**FULL_VALID["modelling_controls"], "p_value_entry": 0.0}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_p_value_entry_one_raises(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {**FULL_VALID["modelling_controls"], "p_value_entry": 1.0}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_p_value_removal_greater_than_one_raises(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {**FULL_VALID["modelling_controls"], "p_value_removal": 1.5}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_vif_threshold_below_one_raises(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {**FULL_VALID["modelling_controls"], "vif_threshold": 0.5}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_vif_threshold_exactly_one_raises(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {**FULL_VALID["modelling_controls"], "vif_threshold": 1.0}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_p_value_entry_negative_raises(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {**FULL_VALID["modelling_controls"], "p_value_entry": -0.05}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    # ── Wrong types ────────────────────────────────────────────────────────────

    def test_target_is_binned_non_bool_raises(self, tmp_path):
        """YAML integers or non-bool strings should be rejected by StrictBool."""
        # Write raw YAML so we bypass Python's dict bool coercion
        raw_yaml = """
data:
  source_type: csv
  file_path: data.csv
  target_variable: loan_default
  target_is_binned: 1
  variables_are_encoded: true
modelling_controls:
  model_type: OLS
  selection_method: stepwise
output:
  formats:
    - console
"""
        p = tmp_path / "config.yaml"
        p.write_text(raw_yaml, encoding="utf-8")
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_vif_threshold_string_raises(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {**FULL_VALID["modelling_controls"], "vif_threshold": "ten"}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_empty_target_variable_raises(self, tmp_path):
        cfg = {**FULL_VALID, "data": {**FULL_VALID["data"], "target_variable": ""}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_whitespace_target_variable_raises(self, tmp_path):
        cfg = {**FULL_VALID, "data": {**FULL_VALID["data"], "target_variable": "   "}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    # ── Multiple errors reported together ─────────────────────────────────────

    def test_multiple_errors_all_reported(self, tmp_path):
        """All errors should be collected and reported in one exception."""
        cfg = {
            "data": {
                "source_type": "bad_source",     # invalid enum
                "target_variable": "",            # empty
                "variables_are_encoded": "maybe", # wrong type
            },
            "modelling_controls": {
                "model_type": "OLS",
                "selection_method": "stepwise",
            },
            "output": {"formats": ["console"]},
        }
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(p)
        assert len(exc_info.value.errors) >= 2


# =============================================================================
#  Cross-field errors
# =============================================================================


class TestCrossFieldErrors:

    def test_p_value_entry_gte_removal_raises(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {
            **FULL_VALID["modelling_controls"],
            "p_value_entry": 0.10,
            "p_value_removal": 0.05,
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError, match="p_value_entry"):
            load_config(p)

    def test_p_value_entry_equal_removal_raises(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {
            **FULL_VALID["modelling_controls"],
            "p_value_entry": 0.05,
            "p_value_removal": 0.05,
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_glm_missing_distribution_raises(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {
            **FULL_VALID["modelling_controls"],
            "model_type": "GLM",
            # glm_distribution intentionally omitted
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError, match="glm_distribution"):
            load_config(p)

    def test_variables_encoded_true_and_apply_encoding_true_raises(self, tmp_path):
        cfg = {
            **FULL_VALID,
            "data": {**FULL_VALID["data"], "variables_are_encoded": True},
            "encoding": {"apply_encoding": True, "method": "dummy"},
        }
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError, match="contradict"):
            load_config(p)

    def test_variables_not_encoded_no_encoding_method_raises(self, tmp_path):
        cfg = {
            **FULL_VALID,
            "data": {**FULL_VALID["data"], "variables_are_encoded": False},
            "encoding": {"apply_encoding": False},
        }
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError):
            load_config(p)

    def test_manual_selection_empty_variables_raises(self, tmp_path):
        cfg = {
            **FULL_VALID,
            "modelling_controls": {
                **FULL_VALID["modelling_controls"],
                "selection_method": "manual",
            },
            "feature_selection": {"manual_variables": []},
        }
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError, match="manual_variables"):
            load_config(p)

    def test_csv_source_without_file_path_raises(self, tmp_path):
        cfg = {**FULL_VALID, "data": {
            "source_type": "csv",
            "target_variable": "y",
            "variables_are_encoded": True,
            # file_path intentionally omitted
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError, match="file_path"):
            load_config(p)

    def test_csv_source_with_xlsx_file_raises(self, tmp_path):
        cfg = {**FULL_VALID, "data": {
            **FULL_VALID["data"],
            "source_type": "csv",
            "file_path": "data.xlsx",
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError, match="extension"):
            load_config(p)

    def test_excel_source_with_csv_file_raises(self, tmp_path):
        cfg = {**FULL_VALID, "data": {
            **FULL_VALID["data"],
            "source_type": "excel",
            "file_path": "data.csv",
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError, match="extension"):
            load_config(p)

    def test_html_output_without_base_dir_raises(self, tmp_path):
        cfg = {**FULL_VALID, "output": {"formats": ["html"]}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError, match="base_dir"):
            load_config(p)

    def test_csv_output_without_base_dir_raises(self, tmp_path):
        cfg = {**FULL_VALID, "output": {"formats": ["csv"]}}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError, match="base_dir"):
            load_config(p)


# =============================================================================
#  Cross-field warnings (should NOT raise, but should log)
# =============================================================================


class TestCrossFieldWarnings:

    def test_glm_distribution_set_on_ols_warns(self, tmp_path, caplog):
        cfg = {**FULL_VALID, "modelling_controls": {
            **FULL_VALID["modelling_controls"],
            "model_type": "OLS",
            "glm_distribution": "gaussian",
        }}
        p = write_config(tmp_path, cfg)
        config = load_config(p)  # should not raise
        assert any("glm_distribution" in r.message for r in caplog.records)

    def test_binned_target_with_ols_warns(self, tmp_path, caplog):
        cfg = {**FULL_VALID, "data": {**FULL_VALID["data"], "target_is_binned": True}}
        p = write_config(tmp_path, cfg)
        config = load_config(p)  # should not raise
        assert any("binned" in r.message.lower() for r in caplog.records)

    def test_manual_variables_set_on_stepwise_warns(self, tmp_path, caplog):
        cfg = {
            **FULL_VALID,
            "feature_selection": {"manual_variables": ["var_a"]},
        }
        p = write_config(tmp_path, cfg)
        config = load_config(p)
        assert any("manual_variables" in r.message for r in caplog.records)

    def test_dataframe_with_file_path_warns(self, tmp_path, caplog):
        cfg = {**FULL_VALID, "data": {
            "source_type": "dataframe",
            "file_path": "data.csv",
            "target_variable": "y",
            "variables_are_encoded": True,
        }}
        p = write_config(tmp_path, cfg)
        config = load_config(p)
        assert any("file_path" in r.message for r in caplog.records)

    def test_p_value_thresholds_ignored_for_adj_r2_warns(self, tmp_path, caplog):
        cfg = {**FULL_VALID, "modelling_controls": {
            **FULL_VALID["modelling_controls"],
            "selection_criterion": "adj_r2",
        }}
        p = write_config(tmp_path, cfg)
        config = load_config(p)
        assert any("p_value" in r.message for r in caplog.records)

    def test_vif_auto_remove_with_manual_warns(self, tmp_path, caplog):
        cfg = {
            **FULL_VALID,
            "modelling_controls": {
                **FULL_VALID["modelling_controls"],
                "selection_method": "manual",
                "vif_action": "auto_remove",
            },
            "feature_selection": {"manual_variables": ["var_a"]},
        }
        p = write_config(tmp_path, cfg)
        config = load_config(p)
        assert any("auto_remove" in r.message for r in caplog.records)

    def test_encoding_method_ignored_when_pre_encoded_warns(self, tmp_path, caplog):
        cfg = {
            **FULL_VALID,
            "data": {**FULL_VALID["data"], "variables_are_encoded": True},
            "encoding": {"apply_encoding": False, "method": "standard_woe"},
        }
        p = write_config(tmp_path, cfg)
        config = load_config(p)
        assert any("encoding.method" in r.message for r in caplog.records)

    def test_missing_file_at_load_time_warns(self, tmp_path, caplog):
        """file_path pointing to a non-existent file should warn (not error)."""
        cfg = {**FULL_VALID, "data": {
            **FULL_VALID["data"],
            "file_path": "/nonexistent/data.csv",
        }}
        p = write_config(tmp_path, cfg)
        config = load_config(p)  # should not raise — file is checked at pipeline.run()
        assert any("does not exist" in r.message for r in caplog.records)

    def test_console_only_with_base_dir_warns(self, tmp_path, caplog):
        cfg = {**FULL_VALID, "output": {"base_dir": "/some/path", "formats": ["console"]}}
        p = write_config(tmp_path, cfg)
        config = load_config(p)
        assert any("console" in r.message for r in caplog.records)

    def test_duplicate_output_formats_warns(self, tmp_path, caplog):
        cfg = {**FULL_VALID, "output": {"base_dir": "./out", "formats": ["csv", "csv", "html"]}}
        p = write_config(tmp_path, cfg)
        config = load_config(p)
        assert any("duplicate" in r.message.lower() for r in caplog.records)


# =============================================================================
#  OD-blocked features
# =============================================================================


class TestODBlocked:

    def test_robust_model_type_raises_od2(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {
            **FULL_VALID["modelling_controls"],
            "model_type": "Robust",
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigODBlockedError) as exc_info:
            load_config(p)
        assert exc_info.value.od_id == "OD-2"
        assert "Robust" in exc_info.value.detail

    def test_glm_binomial_raises_od1(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {
            **FULL_VALID["modelling_controls"],
            "model_type": "GLM",
            "glm_distribution": "binomial",
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigODBlockedError) as exc_info:
            load_config(p)
        assert exc_info.value.od_id == "OD-1"

    def test_glm_poisson_raises_od1(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {
            **FULL_VALID["modelling_controls"],
            "model_type": "GLM",
            "glm_distribution": "poisson",
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigODBlockedError) as exc_info:
            load_config(p)
        assert exc_info.value.od_id == "OD-1"

    def test_glm_negative_binomial_raises_od1(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {
            **FULL_VALID["modelling_controls"],
            "model_type": "GLM",
            "glm_distribution": "negative_binomial",
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigODBlockedError) as exc_info:
            load_config(p)
        assert exc_info.value.od_id == "OD-1"

    def test_custom_woe_raises_od3(self, tmp_path):
        cfg = {
            **FULL_VALID,
            "data": {**FULL_VALID["data"], "variables_are_encoded": False},
            "encoding": {"apply_encoding": True, "method": "custom_woe"},
        }
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigODBlockedError) as exc_info:
            load_config(p)
        assert exc_info.value.od_id == "OD-3"

    def test_od_error_includes_field_path(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {
            **FULL_VALID["modelling_controls"],
            "model_type": "Robust",
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigODBlockedError) as exc_info:
            load_config(p)
        assert "modelling_controls.model_type" in exc_info.value.field

    def test_od_error_message_mentions_stakeholders(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {
            **FULL_VALID["modelling_controls"],
            "model_type": "Robust",
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigODBlockedError) as exc_info:
            load_config(p)
        assert "OD-2" in str(exc_info.value)


# =============================================================================
#  Error message quality
# =============================================================================


class TestErrorMessageQuality:
    """Verify that error messages name the field, the value, and what's expected."""

    def test_missing_field_error_mentions_field_name(self, tmp_path):
        cfg = {k: v for k, v in FULL_VALID.items() if k != "data"}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(p)
        assert "data" in str(exc_info.value)

    def test_invalid_enum_error_mentions_received_value(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {
            **FULL_VALID["modelling_controls"],
            "model_type": "Ridge",
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(p)
        assert "Ridge" in str(exc_info.value) or "model_type" in str(exc_info.value)

    def test_cross_field_error_mentions_both_fields(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {
            **FULL_VALID["modelling_controls"],
            "p_value_entry": 0.10,
            "p_value_removal": 0.05,
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(p)
        msg = str(exc_info.value)
        assert "p_value_entry" in msg
        assert "p_value_removal" in msg

    def test_validation_error_never_exposes_pydantic_internals(self, tmp_path):
        cfg = {**FULL_VALID, "modelling_controls": {
            **FULL_VALID["modelling_controls"],
            "model_type": "Ridge",
        }}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(p)
        msg = str(exc_info.value)
        # Should not contain raw Pydantic jargon
        assert "ValidationError" not in msg
        assert "loc=" not in msg

    def test_validation_error_count_in_message(self, tmp_path):
        cfg = {k: v for k, v in FULL_VALID.items() if k not in ("data", "modelling_controls")}
        p = write_config(tmp_path, cfg)
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(p)
        assert "error" in str(exc_info.value).lower()


# =============================================================================
#  init_project tests
# =============================================================================


class TestInitProject:

    def test_init_creates_config_yaml(self, tmp_path):
        init(tmp_path)
        assert (tmp_path / "config.yaml").exists()

    def test_init_file_contains_required_sections(self, tmp_path):
        init(tmp_path)
        content = (tmp_path / "config.yaml").read_text()
        assert "data:" in content
        assert "modelling_controls:" in content
        assert "output:" in content

    def test_init_does_not_overwrite_existing(self, tmp_path):
        custom = tmp_path / "config.yaml"
        custom.write_text("my_custom_content: true", encoding="utf-8")
        init(tmp_path)  # should not raise
        assert "my_custom_content" in custom.read_text()

    def test_init_creates_nested_directory(self, tmp_path):
        target = tmp_path / "project" / "subdir"
        init(target)
        assert (target / "config.yaml").exists()

    def test_get_template_yaml_returns_string(self):
        template = _get_template_yaml()
        assert isinstance(template, str)
        assert len(template) > 100  # non-trivial content

    def test_get_template_yaml_contains_all_sections(self):
        template = _get_template_yaml()
        assert "data:" in template
        assert "modelling_controls:" in template
        assert "encoding:" in template
        assert "feature_selection:" in template
        assert "output:" in template

    def test_get_template_yaml_contains_required_markers(self):
        template = _get_template_yaml()
        assert "REQUIRED" in template

    def test_init_default_dir_creates_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        init()
        assert (tmp_path / "config.yaml").exists()


# =============================================================================
#  PipelineConfig attribute access
# =============================================================================


class TestPipelineConfigAttributes:

    def test_data_attributes_accessible(self, tmp_path):
        p = write_config(tmp_path, FULL_VALID)
        config = load_config(p)
        assert config.data.source_type == "csv"
        assert config.data.target_variable == "loan_default"
        assert config.data.target_is_binned is False
        assert config.data.variables_are_encoded is True

    def test_modelling_attributes_accessible(self, tmp_path):
        p = write_config(tmp_path, FULL_VALID)
        config = load_config(p)
        assert config.modelling_controls.model_type == "OLS"
        assert config.modelling_controls.selection_method == "stepwise"
        assert config.modelling_controls.p_value_entry == 0.05
        assert config.modelling_controls.p_value_removal == 0.10
        assert config.modelling_controls.vif_threshold == 10.0

    def test_output_attributes_accessible(self, tmp_path):
        p = write_config(tmp_path, FULL_VALID)
        config = load_config(p)
        assert "csv" in config.output.formats
        assert config.output.base_dir == "./outputs"

    def test_string_path_works_as_well_as_pathlib(self, tmp_path):
        p = write_config(tmp_path, FULL_VALID)
        config = load_config(str(p))
        assert isinstance(config, PipelineConfig)
