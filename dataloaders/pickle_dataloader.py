import pickle
from typing import Dict, Tuple, Sequence
from dataloaders.base import BaseDataLoader


class PickleDataLoader(BaseDataLoader):
    def __init__(self, path: str) -> None:
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def get_corpus(self, **kwargs) -> Tuple[
        Dict[int, str],  # format: {finding_id : "finding_text"}
        Dict[
            Tuple[int, str],  # format: {(collection_id, "collection_title") : [finding_ids]}
            Sequence[int]
        ]
    ]:
        corpus, labels = self.data
        return dict(corpus), dict(labels)
