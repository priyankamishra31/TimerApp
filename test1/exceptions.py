"""
exceptions.py
-------------
All custom exceptions for LinearRegressionTool.

Every exception includes a human-readable message that names the exact
field or context. Raw library exceptions (Pydantic, PyYAML, etc.) are
always caught and re-raised as one of these types — analysts never see
an internal stack trace.

Exception hierarchy
-------------------
LRTBaseError                       Base for all package exceptions
├── ConfigError                    Base for all config exceptions
│   ├── ConfigNotFoundError        Path did not exist — template was written
│   ├── ConfigNotConfiguredError   Template placeholders / empty file detected
│   ├── ConfigFileError            File-level problems (encoding, perms, YAML syntax)
│   ├── ConfigValidationError      Field-level or cross-field validation failures
│   └── ConfigODBlockedError       Feature blocked by an open design decision
├── DataLoadError                  Could not load the data file  [stubbed]
├── SchemaError                    Column / dtype validation failure  [stubbed]
└── ScoringError                   Column mismatch when scoring  [stubbed]
"""


# ── Base ───────────────────────────────────────────────────────────────────────


class LRTBaseError(Exception):
    """Base class for every LinearRegressionTool exception.

    Catch this to catch any LRT error.
    Catch a subclass to catch only that category.
    """


# ── Config exceptions ──────────────────────────────────────────────────────────


class ConfigError(LRTBaseError):
    """Base class for all configuration-related errors."""


class ConfigNotFoundError(ConfigError):
    """
    Raised when the supplied config path does not exist.

    A template config has been written to that path. The analyst must
    edit it and re-run — the pipeline cannot start with template values.
    """


class ConfigNotConfiguredError(ConfigError):
    """
    Raised when the config file exists but is clearly not configured:
      - File is empty or contains only whitespace.
      - File contains unedited template sentinel values
        (e.g. target_variable: YOUR_TARGET_HERE).
      - All required top-level sections are absent.

    The pipeline must never start in this state.
    """


class ConfigFileError(ConfigError):
    """
    Raised for file-level problems that prevent reading the config at all:
      - Permission denied
      - File encoding is not UTF-8
      - YAML syntax error
      - Root element is not a YAML mapping (dict)
      - Wrong file extension (with detail on what was found)
    """


class ConfigValidationError(ConfigError):
    """
    Raised when one or more config fields fail validation.

    All field-level AND cross-field errors are collected in a single
    pass so the analyst sees every problem at once.

    Attributes
    ----------
    errors : list[str]
        Human-readable error strings. Each includes the full YAML path,
        the received value, and what was expected.
    warnings : list[str]
        Non-fatal issues that were logged but did not block loading.
        Stored here for inspection; also emitted via the logger.
    """

    def __init__(self, errors: list[str], warnings: list[str] | None = None) -> None:
        self.errors = errors
        self.warnings = warnings or []
        lines = [
            "",
            "=" * 70,
            "  LinearRegressionTool — Config Validation Failed",
            "=" * 70,
            f"  {len(errors)} error(s) found. Fix all of them before running.",
            "-" * 70,
        ]
        for i, err in enumerate(errors, start=1):
            lines.append(f"  [{i}] {err}")
        lines.append("=" * 70)
        super().__init__("\n".join(lines))


class ConfigODBlockedError(ConfigError):
    """
    Raised when a setting requires an Open Decision to be resolved first.

    This is distinct from a validation error — the config is syntactically
    correct but the requested feature is not yet implemented pending a
    team / stakeholder decision.

    Attributes
    ----------
    od_id : str
        The open decision identifier, e.g. "OD-1".
    field : str
        The YAML field path that triggered this, e.g.
        "modelling_controls.glm_distribution".
    detail : str
        Plain-English explanation of what is blocked and why.
    """

    def __init__(self, od_id: str, field: str, detail: str) -> None:
        self.od_id = od_id
        self.field = field
        self.detail = detail
        super().__init__(
            f"\n"
            f"{'=' * 70}\n"
            f"  LinearRegressionTool — Feature Not Yet Available ({od_id})\n"
            f"{'=' * 70}\n"
            f"  Field   : {field}\n"
            f"  Reason  : {detail}\n"
            f"  Status  : Pending resolution of {od_id} with stakeholders.\n"
            f"{'=' * 70}"
        )


# ── Data exceptions  (stubbed — implemented in data_handler sprint) ────────────


class DataLoadError(LRTBaseError):
    """Raised when a data file cannot be loaded."""


class SchemaError(LRTBaseError):
    """Raised when the loaded DataFrame does not match the expected schema."""


# ── Scoring exceptions  (stubbed — implemented in scorer sprint) ───────────────


class ScoringError(LRTBaseError):
    """Raised when new data cannot be scored due to column mismatch."""
