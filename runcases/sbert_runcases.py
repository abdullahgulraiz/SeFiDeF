import corpus_formats
import dataloaders
import techniques
from typing import Sequence
from .base import RunCase


def sbert_zap_arachni_runcases(ds_path: str) -> Sequence[RunCase]:
    dataloader = dataloaders.SefilaDataLoaderV1(ds_path)
    # initialize technique for different embedders
    for embedder in techniques.SbertSemanticSearch.EMBEDDERS[0:1]:
        yield RunCase(
            title=f"SbertSemanticSearch {embedder}",
            dataloader=dataloader,
            corpus_format=corpus_formats.zap_arachni.name_description_solution_1,
            technique=techniques.SbertSemanticSearch(embedder=embedder),
            technique_kwargs=[{"threshold": 0.5}, {"threshold": 0.7}]
        )


def sbert_trivy_anchore_runcases(ds_path: str) -> Sequence[RunCase]:
    # dataloader = dataloaders.SefilaDataLoaderV2(ds_path)
    dataloader = dataloaders.PickleDataLoader(ds_path)
    corpus_format = corpus_formats.static_tools.anchore_trivy_name_title_description
    # initialize technique for different embedders
    for embedder in techniques.SbertSemanticSearch.EMBEDDERS[0:1]:
        yield RunCase(
            title=f"SbertSemanticSearch {embedder}",
            dataloader=dataloader,
            corpus_format=corpus_format,
            technique=techniques.SbertSemanticSearch(embedder=embedder),
            technique_kwargs=[{"threshold": 0.5}, {"threshold": 0.7}]
        )
