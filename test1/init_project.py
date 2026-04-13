"""
init_project.py
---------------
Project scaffolding utility for LinearRegressionTool.

Public API
----------
init(output_dir)  →  None

When an analyst installs the package and wants to create a new config
they call:

    from linear_regression_tool import init
    init()                        # writes config.yaml to the current directory
    init("path/to/project/")      # writes to a specific directory

The function writes a fully commented config.yaml template and stops —
the analyst must edit it before running the pipeline.

This module is NOT part of the pipeline run. It is a one-time setup
helper. It has no imports from other LRT modules except utils and
exceptions to keep its dependency footprint minimal.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from linear_regression_tool.exceptions import ConfigFileError
from linear_regression_tool.utils import get_logger

_log = get_logger(__name__)

# Path to the bundled example config template, relative to this file.
# Resolved at import time so it fails loudly on a broken installation.
_TEMPLATE_PATH = (
    Path(__file__).parent.parent.parent / "config" / "example_config.yaml"
)


def init(output_dir: str | Path = ".") -> None:
    """Write a starter config.yaml to *output_dir*.

    If a config.yaml already exists at the target path the analyst is
    warned and the file is NOT overwritten — their existing config is
    always preserved.

    Parameters
    ----------
    output_dir:
        Directory to write config.yaml into. Created automatically if it
        does not exist. Defaults to the current working directory.

    Raises
    ------
    ConfigFileError
        If the template cannot be found (broken installation) or the
        target directory cannot be written to.
    """
    output_dir = Path(output_dir).resolve()
    target = output_dir / "config.yaml"

    # ── Locate the bundled template ────────────────────────────────────────────
    template_path = _find_template()

    # ── Create the output directory if needed ─────────────────────────────────
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigFileError(
            f"Could not create directory '{output_dir}': {exc}. "
            "Check that you have write permission for this path."
        ) from exc

    # ── Guard: do not overwrite an existing config ────────────────────────────
    if target.exists():
        _log.warning(
            "Config file already exists at '%s'. "
            "It has NOT been overwritten. "
            "Delete the file manually if you want to reset to the template.",
            target,
        )
        return

    # ── Copy template ─────────────────────────────────────────────────────────
    try:
        shutil.copy2(template_path, target)
    except OSError as exc:
        raise ConfigFileError(
            f"Could not write config template to '{target}': {exc}. "
            "Check that you have write permission for this directory."
        ) from exc

    _log.info(
        "\n"
        "%s\n"
        "  LinearRegressionTool — Config Template Created\n"
        "%s\n"
        "  Template written to: %s\n\n"
        "  Next steps:\n"
        "    1. Open the file and edit every field marked  ← REQUIRED\n"
        "    2. Review the optional fields and adjust as needed\n"
        "    3. Run:\n"
        "         from linear_regression_tool import RegressionPipeline\n"
        "         pipeline = RegressionPipeline('%s')\n"
        "         pipeline.run()\n"
        "%s",
        "=" * 70,
        "=" * 70,
        target,
        target,
        "=" * 70,
    )


def _find_template() -> Path:
    """Locate the bundled example_config.yaml.

    Checks the development layout first (repo root/config/) then the
    installed package layout.

    Raises
    ------
    ConfigFileError
        If the template cannot be found in either location.
    """
    # Development / editable install layout
    dev_path = Path(__file__).parent.parent.parent / "config" / "example_config.yaml"
    if dev_path.exists():
        return dev_path

    # Installed package layout (template copied into package directory)
    pkg_path = Path(__file__).parent / "_template_config.yaml"
    if pkg_path.exists():
        return pkg_path

    raise ConfigFileError(
        "Could not locate the bundled config template. "
        "The package installation may be incomplete. "
        "Re-install with:  pip install linear-regression-tool\n"
        f"Searched:\n  {dev_path}\n  {pkg_path}"
    )


def _get_template_yaml() -> str:
    """Return the raw text of the bundled config template.

    Separated from init() so the template content can be inspected and
    tested in isolation without touching the file system.

    Returns
    -------
    str
        The full annotated YAML template as a string.

    Raises
    ------
    ConfigFileError
        If the template file cannot be found or read.
    """
    template_path = _find_template()
    try:
        return template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigFileError(
            f"Could not read the config template at '{template_path}': {exc}"
        ) from exc
