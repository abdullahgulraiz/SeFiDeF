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


def sbert_trivy_anchore_runcases_1(ds_path: str) -> Sequence[RunCase]:
    # dataloader = dataloaders.SefilaDataLoaderV2(ds_path)
    dataloader = dataloaders.PickleDataLoader(ds_path)
    corpus_format = corpus_formats.static_tools.anchore_trivy_description
    # initialize technique for different embedders
    for embedder in techniques.SbertSemanticSearch.EMBEDDERS[0:1]:
        yield RunCase(
            title=f"SbertSemanticSearch {embedder}",
            dataloader=dataloader,
            corpus_format=corpus_format,
            technique=techniques.SbertSemanticSearch(embedder=embedder),
            technique_kwargs=[
                {"threshold": 0.5},
                # {"threshold": 0.7}
            ]
        )


def sbert_trivy_anchore_runcases_2(ds_path: str) -> Sequence[RunCase]:
    # dataloader = dataloaders.SefilaDataLoaderV2(
    #     path=ds_path,
    #     remove_stopwords=False,
    #     remove_linebreaks=False,
    #     remove_special_characters=False
    # )
    dataloader = dataloaders.PickleDataLoader(ds_path)
    corpus_format = corpus_formats.static_tools.anchore_trivy_package_name_cve_id_description
    # initialize technique for different embedders
    for embedder in techniques.SbertSemanticSearch.EMBEDDERS[0:1]:
        yield RunCase(
            title=f"SbertSemanticSearch {embedder}",
            dataloader=dataloader,
            corpus_format=corpus_format,
            technique=techniques.SbertSemanticSearch(embedder=embedder),
            technique_kwargs=[
                {"threshold": 0.8},
                # {"threshold": 0.7}
            ]
        )


def sbert_multiple_static_tools_descriptions(ds_path: str) -> Sequence[RunCase]:
    dataloader = dataloaders.SefilaDataLoaderV2(
        path=ds_path,
        remove_stopwords=False,
        remove_linebreaks=True,
        remove_special_characters=False
    )
    corpus_format = corpus_formats.static_tools.multiple_static_tools_ds_descriptions
    # initialize technique for different embedders
    for embedder in techniques.SbertSemanticSearch.EMBEDDERS[0:1]:
        yield RunCase(
            title=f"SbertSemanticSearch {embedder}",
            dataloader=dataloader,
            corpus_format=corpus_format,
            technique=techniques.SbertSemanticSearch(embedder=embedder),
            technique_kwargs=[
                {"threshold": 0.8, "transitive_clustering": True},
                # {"threshold": 0.7}
            ]
        )
