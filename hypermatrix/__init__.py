"""
HyperMatrix: a federation of heterogeneous, growable sub-networks.

Core idea: instead of one monolithic network trained on a unified loss, the
system is a network of networks. Novel tasks spawn new "stem" clusters that
differentiate from context. Inter-cluster relations strengthen with use.
"""

from .stem import StemCluster
from .relation import Relation
from .novelty import NoveltyDetector
from .federation import Federation

__all__ = ["StemCluster", "Relation", "NoveltyDetector", "Federation"]
