"""
config_loader.py
----------------
Reads and validates the pipeline configuration file (config.yaml).

Public API
----------
load_config(path)  →  PipelineConfig

The function never lets a raw Pydantic / PyYAML / OS exception reach
the caller.  Every failure is caught, converted to a plain-English
message, and re-raised as the appropriate LRT exception type.

Validation is performed in three sequential passes:

  Pass 1 — File-level checks
    Can we open the file? Is it UTF-8? Is it valid YAML?
    Is the root element a dict? Does it look configured?

  Pass 2 — Field-level checks  (Pydantic)
    Are all required fields present and of the correct type?
    Do enum fields contain valid values?
    Are numeric fields within permitted ranges?

  Pass 3 — Cross-field & holistic checks
    Do the settings make sense together?
    Are OD-blocked features requested?
    Are there combinations that contradict each other?
    Are there redundant settings that should warn the user?

All errors from passes 2 and 3 are collected before raising so the
analyst sees every problem in one go.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError
from pydantic import StrictBool

from linear_regression_tool.exceptions import (
    ConfigFileError,
    ConfigNotConfiguredError,
    ConfigNotFoundError,
    ConfigODBlockedError,
    ConfigValidationError,
)
from linear_regression_tool.utils import get_logger

_log = get_logger(__name__)

# ── Sentinel values that indicate an unedited template ────────────────────────
# If any of these appear verbatim in the loaded YAML we know the user has
# not edited the template.
_SENTINELS: frozenset[str] = frozenset(
    {
        "YOUR_TARGET_HERE",
        "YOUR_SOURCE_TYPE_HERE",
        "/path/to/your/data.csv",
    }
)

# ── Valid output format tokens ─────────────────────────────────────────────────
OutputFormat = Literal["csv", "excel", "html", "console", "log"]


# =============================================================================
#  Pydantic sub-models
# =============================================================================


class DataConfig(BaseModel):
    """Validates the [data] section of config.yaml."""

    source_type: Literal["csv", "excel", "dataframe"]
    file_path: str | None = None
    target_variable: str
    target_is_binned: StrictBool = False
    variables_are_encoded: StrictBool

    @field_validator("target_variable")
    @classmethod
    def target_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("target_variable cannot be empty or whitespace.")
        if v.strip() in _SENTINELS:
            raise ValueError(
                "target_variable still contains the template placeholder value "
                f"'{v}'. Edit this field to your actual target column name."
            )
        return v.strip()

    @field_validator("file_path")
    @classmethod
    def file_path_not_sentinel(cls, v: str | None) -> str | None:
        if v and v.strip() in _SENTINELS:
            raise ValueError(
                f"file_path still contains the template placeholder value '{v}'. "
                "Set this to the actual path of your data file."
            )
        return v


class EncodingConfig(BaseModel):
    """Validates the [encoding] section of config.yaml."""

    apply_encoding: StrictBool = False
    method: Literal["dummy", "standard_woe", "custom_woe"] = "dummy"
    columns: list[str] = Field(default_factory=list)


class ModellingConfig(BaseModel):
    """Validates the [modelling_controls] section of config.yaml."""

    model_type: Literal["OLS", "GLM", "Robust"]
    glm_distribution: (
        Literal["gaussian", "binomial", "poisson", "negative_binomial"] | None
    ) = None
    selection_method: Literal["stepwise", "forward", "backward", "manual"]
    selection_criterion: Literal["p_value", "aic_bic", "adj_r2", "all"] = "all"
    p_value_entry: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
    p_value_removal: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.10
    vif_threshold: Annotated[float, Field(gt=1.0)] = 10.0
    vif_action: Literal["warn", "auto_remove"] = "warn"

    @field_validator("p_value_entry", "p_value_removal")
    @classmethod
    def pvalue_range(cls, v: float, info: object) -> float:
        # Boundary guard: 0.0 and 1.0 are mathematically invalid thresholds
        if v <= 0.0 or v >= 1.0:
            field = getattr(info, "field_name", "p_value")
            raise ValueError(
                f"modelling_controls.{field} must be strictly between 0 and 1 "
                f"(exclusive). Received: {v}."
            )
        return v


class FeatureSelectionConfig(BaseModel):
    """Validates the [feature_selection] section of config.yaml."""

    manual_variables: list[str] = Field(default_factory=list)


class OutputConfig(BaseModel):
    """Validates the [output] section of config.yaml."""

    base_dir: str | None = None
    formats: list[OutputFormat]

    @field_validator("formats")
    @classmethod
    def formats_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError(
                "output.formats cannot be empty. "
                "Specify at least one format: csv, excel, html, console, log."
            )
        return v


class PipelineConfig(BaseModel):
    """
    Top-level Pydantic model representing the full config.yaml schema.

    Instantiated by load_config() after the YAML has been read.
    All sub-models are validated first; then model_validator runs
    cross-field checks.
    """

    data: DataConfig
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    modelling_controls: ModellingConfig
    feature_selection: FeatureSelectionConfig = Field(
        default_factory=FeatureSelectionConfig
    )
    output: OutputConfig

    # We store collected warnings here so load_config() can log them
    # after successful validation without re-running cross-field logic.
    model_config = {"arbitrary_types_allowed": True}
    _warnings: list[str] = []

    @model_validator(mode="after")
    def cross_field_validation(self) -> "PipelineConfig":
        """
        Cross-field validation runs AFTER all sub-models are valid.
        Collects errors AND warnings into separate lists, then raises
        ConfigValidationError if any errors exist.
        """
        errors: list[str] = []
        warnings: list[str] = []

        mc = self.modelling_controls
        dc = self.data
        enc = self.encoding
        fs = self.feature_selection
        out = self.output

        # ── model_type cross-checks ───────────────────────────────────────────

        if mc.model_type == "GLM" and mc.glm_distribution is None:
            errors.append(
                "modelling_controls.glm_distribution is required when "
                "modelling_controls.model_type is 'GLM'. "
                "Valid values: gaussian, binomial, poisson, negative_binomial."
            )

        if mc.model_type != "GLM" and mc.glm_distribution is not None:
            warnings.append(
                f"modelling_controls.glm_distribution ('{mc.glm_distribution}') "
                f"is ignored when model_type is '{mc.model_type}'."
            )

        if mc.model_type == "OLS" and dc.target_is_binned:
            warnings.append(
                "data.target_is_binned is true but modelling_controls.model_type "
                "is 'OLS'. OLS on a binned target may produce unreliable results. "
                "Consider using model_type: GLM with an appropriate distribution."
            )

        # ── encoding cross-checks ─────────────────────────────────────────────

        if dc.variables_are_encoded and enc.apply_encoding:
            errors.append(
                "data.variables_are_encoded is true AND "
                "encoding.apply_encoding is true — these flags contradict each other. "
                "Set apply_encoding: false when your data is already encoded, or set "
                "variables_are_encoded: false to apply encoding in the pipeline."
            )

        if dc.variables_are_encoded and enc.method != "dummy":
            warnings.append(
                f"encoding.method ('{enc.method}') is ignored when "
                "data.variables_are_encoded is true."
            )

        if not dc.variables_are_encoded and not enc.apply_encoding:
            errors.append(
                "data.variables_are_encoded is false AND "
                "encoding.apply_encoding is false. "
                "The pipeline has no way to handle un-encoded variables. "
                "Either set variables_are_encoded: true (data is pre-encoded) "
                "or set apply_encoding: true and specify encoding.method."
            )

        if not dc.variables_are_encoded and enc.apply_encoding and not enc.method:
            errors.append(
                "encoding.apply_encoding is true but encoding.method is not set. "
                "Valid values: dummy, standard_woe, custom_woe."
            )

        # ── data source cross-checks ──────────────────────────────────────────

        if dc.source_type in ("csv", "excel") and not dc.file_path:
            errors.append(
                f"data.file_path is required when data.source_type is "
                f"'{dc.source_type}'. "
                "Provide the full path to your data file."
            )

        if dc.source_type == "dataframe" and dc.file_path:
            warnings.append(
                f"data.file_path ('{dc.file_path}') is ignored when "
                "data.source_type is 'dataframe'. "
                "The DataFrame is passed in directly at runtime."
            )

        if dc.file_path and dc.source_type == "csv":
            p = Path(dc.file_path)
            if p.suffix.lower() not in (".csv", ""):
                errors.append(
                    f"data.file_path extension '{p.suffix}' does not match "
                    "data.source_type 'csv'. Expected a .csv file."
                )

        if dc.file_path and dc.source_type == "excel":
            p = Path(dc.file_path)
            if p.suffix.lower() not in (".xlsx", ".xls", ""):
                errors.append(
                    f"data.file_path extension '{p.suffix}' does not match "
                    "data.source_type 'excel'. Expected a .xlsx or .xls file."
                )

        # File existence check — only warn at this stage (file may be created
        # programmatically before pipeline.run() is called). Hard error
        # happens inside data_handler when the file is actually opened.
        if dc.file_path and dc.source_type in ("csv", "excel"):
            if not Path(dc.file_path).exists():
                warnings.append(
                    f"data.file_path '{dc.file_path}' does not exist on disk. "
                    "The pipeline will fail when it tries to load data. "
                    "Ensure the file exists before calling pipeline.run()."
                )

        # ── selection method cross-checks ─────────────────────────────────────

        if mc.selection_method == "manual":
            if not fs.manual_variables:
                errors.append(
                    "feature_selection.manual_variables is empty but "
                    "modelling_controls.selection_method is 'manual'. "
                    "Provide at least one variable name in manual_variables."
                )
            if mc.selection_criterion != "all":
                warnings.append(
                    f"modelling_controls.selection_criterion "
                    f"('{mc.selection_criterion}') is ignored when "
                    "selection_method is 'manual'."
                )
            # p-value thresholds are also irrelevant for manual selection
            warnings.append(
                "modelling_controls.p_value_entry and p_value_removal are "
                "ignored when selection_method is 'manual'."
            )

        if mc.selection_method != "manual" and fs.manual_variables:
            warnings.append(
                f"feature_selection.manual_variables {fs.manual_variables} "
                f"is ignored when selection_method is '{mc.selection_method}'. "
                "Set selection_method: manual to use an explicit variable list."
            )

        if mc.selection_criterion in ("adj_r2", "aic_bic") and mc.selection_method != "manual":
            warnings.append(
                f"modelling_controls.p_value_entry and p_value_removal are "
                f"not used when selection_criterion is '{mc.selection_criterion}'. "
                "These thresholds only apply when criterion is 'p_value' or 'all'."
            )

        # ── p-value ordering cross-check ──────────────────────────────────────

        if mc.p_value_entry >= mc.p_value_removal:
            errors.append(
                f"modelling_controls.p_value_entry ({mc.p_value_entry}) must be "
                f"strictly less than p_value_removal ({mc.p_value_removal}). "
                "The entry threshold must be stricter (smaller) than the removal "
                "threshold. Typical values: entry=0.05, removal=0.10."
            )

        # ── VIF + manual selection cross-check ────────────────────────────────

        if mc.vif_action == "auto_remove" and mc.selection_method == "manual":
            warnings.append(
                "modelling_controls.vif_action is 'auto_remove' but "
                "selection_method is 'manual'. VIF auto-removal may silently "
                "remove variables you manually specified. "
                "Consider setting vif_action: warn when using manual selection."
            )

        # ── output cross-checks ───────────────────────────────────────────────

        file_formats = {"csv", "excel", "html", "log"}
        needs_base_dir = file_formats.intersection(set(out.formats))
        if needs_base_dir and not out.base_dir:
            errors.append(
                f"output.base_dir is required when output.formats includes "
                f"{sorted(needs_base_dir)}. "
                "Set base_dir to a writable directory path."
            )

        if out.formats == ["console"] and out.base_dir is not None:
            warnings.append(
                "output.base_dir is set but output.formats only contains "
                "'console'. No files will be written to disk."
            )

        # Duplicate format check
        seen: set[str] = set()
        dupes = [f for f in out.formats if f in seen or seen.add(f)]  # type: ignore[func-returns-value]
        if dupes:
            warnings.append(
                f"output.formats contains duplicate values: {dupes}. "
                "Duplicates will be ignored."
            )

        # ── holistic "nothing makes sense" check ─────────────────────────────

        if errors and len(errors) >= 3:
            # If almost everything is wrong the file is probably still a
            # template — add a helpful top-level hint.
            warnings.insert(
                0,
                "Multiple critical fields are invalid. If you have not yet "
                "edited the generated config template, please do so before "
                "running the pipeline.",
            )

        # ── store warnings and raise if any errors ────────────────────────────

        object.__setattr__(self, "_warnings", warnings)

        if errors:
            raise ConfigValidationError(errors=errors, warnings=warnings)

        return self


# =============================================================================
#  Public function
# =============================================================================


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate the pipeline configuration file.

    If the file does not exist a template is written to *path* and
    ``ConfigNotFoundError`` is raised — the analyst must edit the
    template before re-running.

    Parameters
    ----------
    path:
        Path to the YAML configuration file. Relative paths are resolved
        from the current working directory.

    Returns
    -------
    PipelineConfig
        Fully validated configuration object.

    Raises
    ------
    ConfigNotFoundError
        File did not exist; template has been written to *path*.
    ConfigNotConfiguredError
        File is empty, whitespace-only, or still contains template
        sentinel values.
    ConfigFileError
        File cannot be read (permissions, encoding) or is not valid YAML.
    ConfigValidationError
        One or more fields failed validation (all errors reported at once).
    ConfigODBlockedError
        A setting requires an open design decision to be resolved first.
    """
    path = Path(path).resolve()

    # ── Pass 0: file existence ─────────────────────────────────────────────────
    _check_or_create_config(path)

    # ── Pass 1: file-level read ────────────────────────────────────────────────
    raw = _read_yaml(path)

    # ── Pass 1b: sentinel / empty check ───────────────────────────────────────
    _check_for_sentinels(raw, path)

    # ── Pass 2 + 3: Pydantic field validation + cross-field validation ─────────
    config = _validate(raw, path)

    # ── Post-validation: OD-blocked feature checks ────────────────────────────
    _check_od_blocked(config)

    # ── Emit any accumulated warnings ────────────────────────────────────────
    stored_warnings: list[str] = getattr(config, "_warnings", [])
    for w in stored_warnings:
        _log.warning(w)

    _log.info(
        "Config loaded successfully from '%s'. "
        "model_type=%s  selection_method=%s  target=%s",
        path,
        config.modelling_controls.model_type,
        config.modelling_controls.selection_method,
        config.data.target_variable,
    )
    return config


# =============================================================================
#  Private helpers
# =============================================================================


def _check_or_create_config(path: Path) -> None:
    """Write template if the path does not exist, then raise."""
    if path.exists():
        return

    _log.info(
        "Config file not found at '%s'. Writing template...", path
    )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        template = _load_template()
        path.write_text(template, encoding="utf-8")
    except OSError as exc:
        raise ConfigFileError(
            f"Could not write template config to '{path}': {exc}. "
            "Check that the directory is writable."
        ) from exc

    raise ConfigNotFoundError(
        f"\n"
        f"{'=' * 70}\n"
        f"  LinearRegressionTool — Config Template Created\n"
        f"{'=' * 70}\n"
        f"  A template config file has been written to:\n"
        f"    {path}\n\n"
        f"  You must edit this file before running the pipeline.\n"
        f"  Look for every field marked  ← REQUIRED  and set it to\n"
        f"  your actual values.\n"
        f"{'=' * 70}"
    )


def _load_template() -> str:
    """Return the contents of the bundled example_config.yaml template."""
    template_path = Path(__file__).parent.parent.parent / "config" / "example_config.yaml"
    if not template_path.exists():
        # Fallback: look relative to the installed package
        template_path = Path(__file__).parent / "_template_config.yaml"

    if not template_path.exists():
        raise ConfigFileError(
            "Could not locate the bundled config template. "
            "The package installation may be incomplete. "
            "Re-install with: pip install linear-regression-tool"
        )

    return template_path.read_text(encoding="utf-8")


def _read_yaml(path: Path) -> dict:
    """Read the YAML file and return a raw dict.

    Raises ConfigFileError for any file-level or YAML-level problem.
    """
    # Extension warning (we still attempt to parse)
    if path.suffix.lower() not in (".yaml", ".yml"):
        _log.warning(
            "Config file '%s' has extension '%s'. Expected .yaml or .yml. "
            "Attempting to parse anyway.",
            path,
            path.suffix,
        )

    # Permission / encoding
    try:
        raw_text = path.read_text(encoding="utf-8")
    except PermissionError as exc:
        raise ConfigFileError(
            f"Permission denied reading config file '{path}'. "
            "Check that you have read access to this file."
        ) from exc
    except UnicodeDecodeError as exc:
        raise ConfigFileError(
            f"Config file '{path}' is not valid UTF-8. "
            f"Re-save the file with UTF-8 encoding. Detail: {exc}"
        ) from exc
    except OSError as exc:
        raise ConfigFileError(
            f"Could not read config file '{path}': {exc}"
        ) from exc

    # Empty file
    if not raw_text.strip():
        raise ConfigNotConfiguredError(
            f"Config file '{path}' is empty. "
            "It looks like the template was created but not edited. "
            "Edit the file and re-run."
        )

    # YAML syntax
    try:
        parsed = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        # Extract line/column info from the YAML error if available
        mark = getattr(exc, "problem_mark", None)
        location = (
            f" (line {mark.line + 1}, column {mark.column + 1})"
            if mark
            else ""
        )
        raise ConfigFileError(
            f"Config file '{path}' contains invalid YAML{location}. "
            f"Detail: {_clean_yaml_error(exc)}"
        ) from exc

    # Root must be a mapping
    if not isinstance(parsed, dict):
        raise ConfigFileError(
            f"Config file '{path}' is valid YAML but the root element is a "
            f"{type(parsed).__name__}, not a mapping (dict). "
            "The config file must start with top-level keys like 'data:',  "
            "'modelling_controls:', etc."
        )

    return parsed


def _check_for_sentinels(raw: dict, path: Path) -> None:
    """Recursively scan the parsed YAML for unedited template values."""
    found: list[str] = []
    _scan_for_sentinels(raw, prefix="", found=found)

    if found:
        fields = "\n".join(f"  • {f}" for f in found)
        raise ConfigNotConfiguredError(
            f"\n"
            f"{'=' * 70}\n"
            f"  LinearRegressionTool — Config Not Configured\n"
            f"{'=' * 70}\n"
            f"  The following fields in '{path}' still contain\n"
            f"  template placeholder values:\n"
            f"{fields}\n\n"
            f"  Edit these fields with your actual values and re-run.\n"
            f"{'=' * 70}"
        )


def _scan_for_sentinels(
    node: object, prefix: str, found: list[str]
) -> None:
    """Recursively walk *node* looking for sentinel strings."""
    if isinstance(node, dict):
        for k, v in node.items():
            child_prefix = f"{prefix}.{k}" if prefix else k
            _scan_for_sentinels(v, child_prefix, found)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _scan_for_sentinels(item, f"{prefix}[{i}]", found)
    elif isinstance(node, str) and node.strip() in _SENTINELS:
        found.append(f"{prefix}: '{node}'")


def _validate(raw: dict, path: Path) -> PipelineConfig:
    """Run Pydantic validation and cross-field checks.

    Converts all Pydantic ValidationErrors into ConfigValidationError
    with human-readable messages.
    """
    try:
        return PipelineConfig(**raw)
    except PydanticValidationError as exc:
        errors = _format_pydantic_errors(exc)
        raise ConfigValidationError(errors=errors) from exc
    except ConfigValidationError:
        raise  # already formatted by model_validator, pass through
    except TypeError as exc:
        # Happens when a top-level section is present but not a dict
        # (e.g. `data: null`)
        raise ConfigFileError(
            f"Config file '{path}' has an unexpected structure: {exc}. "
            "Ensure every section (data:, modelling_controls:, etc.) "
            "is a YAML mapping with key: value pairs, not null or a list."
        ) from exc


def _check_od_blocked(config: PipelineConfig) -> None:
    """Raise ConfigODBlockedError for any setting blocked by an open decision."""
    mc = config.modelling_controls
    enc = config.encoding

    if mc.model_type == "Robust":
        raise ConfigODBlockedError(
            od_id="OD-2",
            field="modelling_controls.model_type",
            detail=(
                "Robust regression (Huber M-estimation) is not yet implemented. "
                "This feature is pending resolution of OD-2 "
                "(include in v1.0 or defer to v1.1). "
                "Use model_type: OLS or model_type: GLM for now."
            ),
        )

    if mc.model_type == "GLM" and mc.glm_distribution and mc.glm_distribution != "gaussian":
        raise ConfigODBlockedError(
            od_id="OD-1",
            field="modelling_controls.glm_distribution",
            detail=(
                f"GLM distribution '{mc.glm_distribution}' is not yet active. "
                "Only 'gaussian' (identity link) is implemented in v1.0. "
                "Additional distributions are pending resolution of OD-1 "
                "(which GLM distributions / link functions are needed). "
                "Set glm_distribution: gaussian to use GLM now."
            ),
        )

    if enc.method == "custom_woe":
        raise ConfigODBlockedError(
            od_id="OD-3",
            field="encoding.method",
            detail=(
                "Custom linear-target WoE encoding is not yet implemented. "
                "This feature is pending receipt of the insurance WoE "
                "documentation (OD-3). "
                "Use encoding.method: standard_woe or dummy for now."
            ),
        )


# =============================================================================
#  Error formatting helpers
# =============================================================================


def _format_pydantic_errors(exc: PydanticValidationError) -> list[str]:
    """Convert a Pydantic ValidationError into plain-English strings.

    Each string includes the YAML field path, what was received, and
    what was expected — no internal Pydantic jargon.
    """
    messages: list[str] = []

    for error in exc.errors():
        # Build a dot-separated YAML path from the Pydantic loc tuple
        loc_parts = [str(p) for p in error.get("loc", [])]
        field_path = ".".join(loc_parts) if loc_parts else "unknown field"

        raw_msg = error.get("msg", "")
        error_type = error.get("type", "")
        raw_input = error.get("input")

        message = _humanise_pydantic_error(
            field_path=field_path,
            raw_msg=raw_msg,
            error_type=error_type,
            raw_input=raw_input,
        )
        messages.append(message)

    return messages


def _humanise_pydantic_error(
    field_path: str,
    raw_msg: str,
    error_type: str,
    raw_input: object,
) -> str:
    """Convert one Pydantic error dict into a readable sentence."""

    # ── Missing required field ────────────────────────────────────────────────
    if error_type == "missing":
        return (
            f"'{field_path}' is required but was not found in the config. "
            "Add this field and set it to a valid value."
        )

    # ── Literal / enum mismatch ───────────────────────────────────────────────
    if error_type in ("literal_error", "enum"):
        expected = _extract_expected_from_msg(raw_msg)
        received = f"'{raw_input}'" if isinstance(raw_input, str) else str(raw_input)
        return (
            f"'{field_path}' has an invalid value {received}. "
            f"Expected one of: {expected}."
        )

    # ── Wrong type ────────────────────────────────────────────────────────────
    if error_type in ("bool_type", "bool_parsing"):
        received = f"'{raw_input}'"
        return (
            f"'{field_path}' must be true or false (boolean). "
            f"Received: {received}."
        )

    if "type" in error_type and "int" in error_type:
        return (
            f"'{field_path}' must be a whole number (integer). "
            f"Received: '{raw_input}'."
        )

    if "type" in error_type and "float" in error_type:
        return (
            f"'{field_path}' must be a decimal number. "
            f"Received: '{raw_input}'."
        )

    if "string" in error_type:
        return (
            f"'{field_path}' must be a text value. "
            f"Received: {type(raw_input).__name__} '{raw_input}'."
        )

    # ── Numeric range ─────────────────────────────────────────────────────────
    if error_type in ("greater_than", "less_than", "greater_than_equal", "less_than_equal"):
        return (
            f"'{field_path}' is out of range. "
            f"Received: {raw_input}. Detail: {raw_msg}."
        )

    # ── Value error (from our own @field_validator functions) ─────────────────
    if error_type == "value_error":
        # Our validators already produce clean messages — pass them through
        # but strip the Pydantic "Value error, " prefix if present.
        clean = re.sub(r"^Value error,\s*", "", raw_msg)
        return clean

    # ── Null / None where not allowed ─────────────────────────────────────────
    if "none" in error_type.lower():
        return (
            f"'{field_path}' cannot be null/empty. "
            "Provide a valid value for this field."
        )

    # ── Fallback ──────────────────────────────────────────────────────────────
    return (
        f"'{field_path}': {raw_msg} "
        f"(received: '{raw_input}')."
    )


def _extract_expected_from_msg(msg: str) -> str:
    """Pull the expected values out of a Pydantic literal error message."""
    # Pydantic v2 literal error looks like:
    # "Input should be 'csv', 'excel' or 'dataframe'"
    match = re.search(r"Input should be (.+)", msg)
    if match:
        return match.group(1).rstrip(".")
    return msg


def _clean_yaml_error(exc: yaml.YAMLError) -> str:
    """Return a short clean string from a PyYAML error."""
    msg = str(exc)
    # Remove YAML's ASCII-art pointer lines (they look noisy in our messages)
    lines = [ln for ln in msg.splitlines() if not ln.strip().startswith("^")]
    return " ".join(lines).strip()
