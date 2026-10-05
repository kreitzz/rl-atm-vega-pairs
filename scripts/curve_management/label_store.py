from functools import lru_cache
from scripts.curve90.labels import LabelStore

class CachedLabelStore(LabelStore):
    """Six immutable blocks, no lingering mappings or global instance cache."""
    def __init__(self,path,shape):
        super().__init__(path,shape)
        self._cached=lru_cache(maxsize=6)(self._immutable)
    def _immutable(self,index):
        result=super().__getitem__(index);result.setflags(write=False);return result
    def __getitem__(self,index):return self._cached(index)
