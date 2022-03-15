from typing import Sequence
from dataloaders.sefila_dataloader import SefilaDataLoader
from techniques.sbert_semantic_search import SbertSemanticSearch
from runcases.base import RunCase
from corpus_formats import zap_arachni


def sbert_zap_arachni_runcases(ds_path: str) -> Sequence[RunCase]:
    dataloader = SefilaDataLoader(ds_path)
    # initialize technique for different embedders
    for embedder in SbertSemanticSearch.EMBEDDERS[0:1]:
        yield RunCase(
            title=f"SbertSemanticSearch {embedder}",
            dataloader=dataloader,
            corpus_format=zap_arachni.name_description_solution_1,
            technique=SbertSemanticSearch(embedder=embedder),
            technique_kwargs=[{"threshold": 0.5}, {"threshold": 0.7}]
        )
