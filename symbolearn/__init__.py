"""symbolearn: symbolic discovery of explicit and auditable spectral models for land cover mapping."""

from .expression import Expression, ExpressionSet
from .fitness import Fitness
from .symbolic_estimators import (
    SymbolicClassifier,
    SymbolicRegressor,
    SymbolicTransformer,
)

# Keep in sync with [project] version in pyproject.toml.
__version__ = "0.1.0"

__all__ = [
    "Expression",
    "ExpressionSet",
    "Fitness",
    "SymbolicClassifier",
    "SymbolicRegressor",
    "SymbolicTransformer",
    "__version__",
]
