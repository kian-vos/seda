"""Analysis module for SEDA."""

from seda.analysis.nlp import PersianNLP
from seda.analysis.features import FeatureExtractor
from seda.analysis.bot import BotDetector
from seda.analysis.stance import StanceClassifier
from seda.analysis.coordination import CoordinationDetector

# Conditional import for embeddings (requires transformers)
try:
    from seda.analysis.embeddings import PersianEmbedder, is_embeddings_available
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

    def is_embeddings_available() -> bool:
        return False

__all__ = [
    "PersianNLP",
    "FeatureExtractor",
    "BotDetector",
    "StanceClassifier",
    "CoordinationDetector",
    "is_embeddings_available",
]

if _EMBEDDINGS_AVAILABLE:
    __all__.append("PersianEmbedder")
