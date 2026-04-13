"""
linear_regression_tool
----------------------
SAS PROC REG / PROC GLM → Python (statsmodels) regression pipeline.

Public API
----------
The only symbol analysts should import directly is RegressionPipeline.
All other modules are internal implementation details.

    from linear_regression_tool import RegressionPipeline, init
    init()                               # create starter config.yaml
    pipeline = RegressionPipeline("config.yaml")
    pipeline.run()

Stage 2 (ipywidgets UI) is installed separately:
    pip install linear-regression-tool[notebook-ui]
"""

from linear_regression_tool.init_project import init
from linear_regression_tool.pipeline import RegressionPipeline

__version__ = "1.0.0"
__all__ = ["RegressionPipeline", "init", "__version__"]
