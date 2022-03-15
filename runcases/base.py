from pprint import pprint
from dataloaders.base import BaseDataLoader
from techniques.base import BaseTechnique
from corpus_formats.base import CorpusFormat
from typing import Union, Sequence


class RunCase:
    def __init__(
            self,
            title: str,
            dataloader: BaseDataLoader,
            corpus_format: CorpusFormat,
            technique: BaseTechnique,
            technique_kwargs: Union[dict, Sequence[dict]] = None
    ):
        self.title = title
        self.dataloader = dataloader
        self.corpus_format = corpus_format
        self.corpus, self.labels = self.dataloader.get_corpus(**corpus_format.format_dict)
        self.technique = technique
        self.technique_kwargs = technique_kwargs

    def execute(self):
        # add empty kwargs if absent
        if not self.technique_kwargs:
            self.technique_kwargs = [{}]
        # convert normal kwargs to sequence
        if not isinstance(self.technique_kwargs, Sequence):
            self.technique_kwargs = [self.technique_kwargs]
        # apply technique on corpus and evaluate results
        for technique_kwargs in self.technique_kwargs:
            print(f"==== \n"
                  f"RunCase: `{self.title}`, "
                  f"Corpus: {self.corpus_format.name}, "
                  f"Params: `{technique_kwargs}` "
                  f"\n====")
            results = self.technique.apply(self.corpus, **technique_kwargs)
            evaluation = self.technique.evaluate(self.labels, results)
            pprint(evaluation)
