"""
utils.py
--------
Shared utilities used by every module in LinearRegressionTool.

Rules
-----
- No module in this package uses print(). Everything goes through
  get_logger() defined here.
- This file must have ZERO package-level dependencies beyond the
  Python standard library. It is imported by every other module —
  a new dependency here is a dependency everywhere.
- Functions beyond get_logger() are stubbed and will be implemented
  in the outputs/scorer sprint.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


# ── Logger ─────────────────────────────────────────────────────────────────────

# Single shared formatter so every module's log lines look identical.
_FORMATTER = logging.Formatter(
    fmt="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# Registry so we never add duplicate handlers when get_logger is called
# multiple times with the same name (common in tests).
_LOGGER_REGISTRY: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for *name*.

    The logger writes INFO-and-above messages to stdout. DEBUG messages
    are suppressed at the handler level unless the root logger is set to
    DEBUG explicitly (e.g. during tests).

    A rotating file handler will be added in the outputs sprint once the
    output directory structure is established. For now all output goes to
    the terminal.

    Parameters
    ----------
    name:
        Typically ``__name__`` from the calling module, e.g.
        ``linear_regression_tool.config_loader``.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if name in _LOGGER_REGISTRY:
        return logger  # already configured — don't add duplicate handlers

    logger.setLevel(logging.DEBUG)  # let handlers decide what to surface

    # Console handler — INFO and above to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_FORMATTER)
    logger.addHandler(console_handler)

    # Allow propagation so pytest's caplog fixture can capture log records
    # during tests. In production the root logger has no handlers by default
    # so records won't be printed twice.
    logger.propagate = True

    _LOGGER_REGISTRY.add(name)
    return logger


# ── Output directory helpers  (stubbed — implemented in outputs sprint) ────────


def ensure_output_dirs(base_dir: str | Path) -> dict[str, Path]:
    """Create the standard output directory structure under *base_dir*.

    Creates: reports/, html/, plots/, logs/, artefacts/

    Returns a dict mapping short name → absolute Path so callers never
    have to construct subdirectory paths manually.

    .. note::
        Stubbed. Full implementation in the outputs sprint.
    """
    raise NotImplementedError(
        "ensure_output_dirs() will be implemented in the outputs sprint."
    )


# ── Formatting helpers  (stubbed — implemented in outputs sprint) ──────────────


def format_p_value(p: float) -> str:
    """Format a p-value for display, mirroring SAS output style.

    Returns '<0.001' for very small values, '0.XXX' otherwise.

    .. note::
        Stubbed. Full implementation in the outputs sprint.
    """
    raise NotImplementedError(
        "format_p_value() will be implemented in the outputs sprint."
    )


def format_table(df: object) -> str:  # df: pd.DataFrame
    """Format a DataFrame as an ASCII table mirroring the SAS log window.

    .. note::
        Stubbed. Full implementation in the outputs sprint.
    """
    raise NotImplementedError(
        "format_table() will be implemented in the outputs sprint."
    )


# ── Timestamp helper  (stubbed — implemented in outputs sprint) ────────────────


def timestamp() -> str:
    """Return an ISO-format timestamp string for log/artefact naming.

    .. note::
        Stubbed. Full implementation in the outputs sprint.
    """
    raise NotImplementedError(
        "timestamp() will be implemented in the outputs sprint."
    )
