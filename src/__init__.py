"""Influence-functions prototype package.

Public API exports for the most commonly used classes. Subclass `AbstractTask`
to plug in your own model and dataset (see `examples/` for reference
implementations), then instantiate one of the `*Computer` classes below.
"""

from src.abstract_computer import AbstractComputer
from src.abstract_task import AbstractTask, InvalidTaskError, validate_task
from src.gradient_similarity import GradientSimilarityComputer
from src.influence_function import InfluenceFunctionComputer
from src.representation_similarity import RepresentationSimilarityComputer
from src.source import SourceComputer
from src.tracin import TracinComputer

__all__ = [
    "AbstractComputer",
    "AbstractTask",
    "GradientSimilarityComputer",
    "InfluenceFunctionComputer",
    "InvalidTaskError",
    "RepresentationSimilarityComputer",
    "SourceComputer",
    "TracinComputer",
    "validate_task",
]
