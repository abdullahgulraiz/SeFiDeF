from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    @abstractmethod
    def __init__(self, path: str):
        pass

    @abstractmethod
    def get_corpus(self, **kwargs):
        pass
