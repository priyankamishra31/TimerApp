"""
pipeline.py
-----------
RegressionPipeline — the single public entry point for the package.

All Stage 1 module calls are orchestrated here. Analysts interact only
with this class. Internal modules are never imported directly.

This file is a STUB. Only __init__() is implemented in the config sprint.
All other methods will be implemented in their respective sprints.

Stage 1 / Stage 2 seam
-----------------------
Stage 2 UI widgets call RegressionPipeline methods only — they never
import data_handler, feature_selector, regression_engine, or any other
internal module. This guarantees a Stage 2 bug can never corrupt Stage 1.
"""

from __future__ import annotations

from pathlib import Path

from linear_regression_tool.config_loader import PipelineConfig, load_config
from linear_regression_tool.utils import get_logger

_log = get_logger(__name__)


class RegressionPipeline:
    """Orchestrates the full SAS → Python linear regression pipeline.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file. If the file does not exist
        a template is written and ``ConfigNotFoundError`` is raised.

    Examples
    --------
    Simplest usage::

        pipeline = RegressionPipeline("config.yaml")
        pipeline.run()

    Step-by-step usage (for debugging or notebooks)::

        pipeline = RegressionPipeline("config.yaml")
        pipeline.load_data()
        pipeline.encode()
        pipeline.run_vif()
        pipeline.select_features()
        pipeline.fit()
        pipeline.export()

    Scoring new data::

        results = pipeline.score(new_df)
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()

        # Load and validate config — raises immediately if anything is wrong.
        # The pipeline never starts in a broken state.
        self.config: PipelineConfig = load_config(self.config_path)

        # State attributes populated by each step method.
        # Declared here so the full pipeline state is visible at a glance.
        self.X = None           # pd.DataFrame — feature matrix
        self.y = None           # pd.Series    — target vector
        self.result = None      # RegressionResult dataclass
        self.selected_variables: list[str] = []
        self.iteration_log = None  # pd.DataFrame — one row per selection step
        self.output_paths: dict[str, Path] = {}

        _log.info(
            "RegressionPipeline initialised. "
            "Config: model_type=%s  selection=%s  target=%s",
            self.config.modelling_controls.model_type,
            self.config.modelling_controls.selection_method,
            self.config.data.target_variable,
        )

    # ── Full pipeline run ──────────────────────────────────────────────────────

    def run(self) -> "RegressionPipeline":
        """Run the full pipeline end-to-end.

        Calls all step methods in the correct order. Returns self so
        results can be inspected after the call.

        .. note::
            Stub — will be implemented once all step modules are built.
        """
        raise NotImplementedError(
            "RegressionPipeline.run() will be implemented in the integration sprint "
            "once all stage modules are complete."
        )

    # ── Step methods (each implemented in its own sprint) ─────────────────────

    def load_data(self) -> "RegressionPipeline":
        """Load data, validate schema, and split into X and y.

        .. note:: Stub — implemented in the data_handler sprint.
        """
        raise NotImplementedError(
            "load_data() will be implemented in the data_handler sprint."
        )

    def encode(self) -> "RegressionPipeline":
        """Apply encoding step (or pass-through if data is pre-encoded).

        .. note:: Stub — implemented in the encoder sprint.
        """
        raise NotImplementedError(
            "encode() will be implemented in the encoder sprint."
        )

    def run_vif(self) -> "RegressionPipeline":
        """Compute VIF and apply warn / auto-remove logic.

        .. note:: Stub — implemented in the vif_handler sprint.
        """
        raise NotImplementedError(
            "run_vif() will be implemented in the vif_handler sprint."
        )

    def select_features(self) -> "RegressionPipeline":
        """Run variable selection (stepwise / forward / backward / manual).

        .. note:: Stub — implemented in the feature_selector sprint.
        """
        raise NotImplementedError(
            "select_features() will be implemented in the feature_selector sprint."
        )

    def fit(self) -> "RegressionPipeline":
        """Fit the regression model on the selected variables.

        .. note:: Stub — implemented in the regression_engine sprint.
        """
        raise NotImplementedError(
            "fit() will be implemented in the regression_engine sprint."
        )

    def export(self) -> "RegressionPipeline":
        """Export all configured output formats and save model artefact.

        .. note:: Stub — implemented in the outputs sprint.
        """
        raise NotImplementedError(
            "export() will be implemented in the outputs sprint."
        )

    def score(self, X_new: object) -> object:
        """Apply fitted coefficients to new data and return predictions.

        .. note:: Stub — implemented in the scorer sprint.
        """
        raise NotImplementedError(
            "score() will be implemented in the scorer sprint."
        )

    def summary(self) -> dict:
        """Return a dict of model metrics, selected variables, and output paths.

        .. note:: Stub — implemented in the outputs sprint.
        """
        raise NotImplementedError(
            "summary() will be implemented in the outputs sprint."
        )
