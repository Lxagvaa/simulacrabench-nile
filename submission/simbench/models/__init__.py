import copy
from dataclasses import replace

from ..data import Encoded
from .common import HEAVY_ITEM_CAP, Marginal, descendants, rank_items, set_deadline
from .latent import FactorVAE, LatentClass
from .nets import MultiTaskNet
from .ordinal import NONSUBSTANTIVE, OrdinalLogit
from .smoothing import Backoff, ProfileMatch
from .tabpfn import TabPFN
from .trees import GBDT, RandomForest


class Direct:
    """Wraps a model so that it predicts the gate sentinel as an ordinary
    answer instead of going through the gate decomposition."""
    direct = True

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name + "_direct"

    @staticmethod
    def _view(enc: Encoded) -> Encoded:
        out = copy.copy(enc)
        out.items = {n: (replace(it, values=it.options, gated=False)
                         if n in enc.predict and it.gated else it)
                     for n, it in enc.items.items()}
        return out

    def fit(self, enc: Encoded):
        self.inner.fit(self._view(enc))
        return self

    def predict(self, enc: Encoded):
        return self.inner.predict(self._view(enc))


__all__ = ["Backoff", "Direct", "FactorVAE", "GBDT", "HEAVY_ITEM_CAP", "LatentClass", "Marginal",
           "MultiTaskNet", "NONSUBSTANTIVE", "OrdinalLogit", "ProfileMatch", "RandomForest",
           "TabPFN", "descendants", "rank_items", "set_deadline"]
