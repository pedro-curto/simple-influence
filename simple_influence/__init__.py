"""Influence-functions prototype package.

Public API exports for the most commonly used classes. Subclass `AbstractTask`
to plug in your own model and dataset (see `examples/` for reference
implementations), then instantiate one of the `*Computer` classes below.
"""

from simple_influence.abstract_computer import AbstractComputer
from simple_influence.abstract_task import AbstractTask, InvalidTaskError, validate_task
from simple_influence.gradient_similarity import GradientSimilarityComputer
from simple_influence.influence_function import InfluenceFunctionComputer
from simple_influence.representation_similarity import RepresentationSimilarityComputer
from simple_influence.source import SourceComputer
from simple_influence.tracin import TracinComputer

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
