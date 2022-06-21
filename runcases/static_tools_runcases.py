import corpus_formats
import dataloaders
import techniques
from typing import Sequence
from .base import RunCase


def _get_dataloader(ds_path: str):
    if ".json" in ds_path:
        return dataloaders.SefilaDataLoaderV2(ds_path)
    elif ".pkl" in ds_path:
        return dataloaders.PickleDataLoader(ds_path)
    else:
        raise NotImplementedError


def equality_comparison_static_tools(unique_ds_path: str, target_ds_path: str) -> Sequence[RunCase]:
    cve_ids_dataloader = _get_dataloader(unique_ds_path)
    descriptions_dataloader = _get_dataloader(target_ds_path)
    unique_corpus_format = corpus_formats.anchore_trivy_cve_id
    dataloader = dataloaders.AggregatedDataloader(cve_ids_dataloader, unique_corpus_format, descriptions_dataloader)
    # dataloader = dataloaders.PickleDataLoader(ds_path)
    descriptions_corpus_format = corpus_formats.anchore_trivy_description
    for embedder in techniques.SbertSemanticSearch.EMBEDDERS[0:1]:
        yield RunCase(
            title=f"Equality Comparison Static Tools, SbertSemanticSearch {embedder}",
            dataloader=dataloader,
            corpus_format=descriptions_corpus_format,
            technique=techniques.SbertSemanticSearch(embedder=embedder),
            technique_kwargs=[
                # {"threshold": 0.3},
                # {"threshold": 0.3, "transitive_clustering": True},
                # {"threshold": 0.5, "transitive_clustering": True},
                # {"threshold": 0.5, "transitive_clustering": False},
                {"threshold": 0.7, "transitive_clustering": True},
                # {"threshold": 0.7, "transitive_clustering": False},
                # {"threshold": 0.8, "transitive_clustering": True},
                # {"threshold": 0.9, "transitive_clustering": True},
                # {"threshold": 0.7}
            ],
            # save_runcase_file_path="/home/abdullah/LRZ Sync+Share/TUM BMC/Master Thesis/Data/runcases_results/"
            #                        "equality_comparison_static_tools.json",
        )


def corpus_aggregation_static_tools_descriptions(unique_ds_path: str, target_ds_path: str) -> Sequence[RunCase]:
    cve_ids_dataloader = _get_dataloader(unique_ds_path)
    descriptions_dataloader = _get_dataloader(target_ds_path)
    unique_corpus_format = corpus_formats.multiple_static_tools_ds_cve_ids
    dataloader = AggregatedDataloader(
        unique_keys_dataloader=cve_ids_dataloader,
        unique_keys_corpus_format=unique_corpus_format,
        target_corpus_dataloader=descriptions_dataloader
    )
    # dataloader = dataloaders.PickleDataLoader(ds_path)
    descriptions_corpus_format = corpus_formats.multiple_static_tools_ds_descriptions
    for embedder in techniques.SbertSemanticSearch.EMBEDDERS[0:1]:
        yield RunCase(
            title=f"Corpus Aggregation, SbertSemanticSearch {embedder}",
            dataloader=dataloader,
            corpus_format=descriptions_corpus_format,
            technique=techniques.SbertSemanticSearch(embedder=embedder),
            technique_kwargs=[
                # {"threshold": 0.3},
                # {"threshold": 0.3, "transitive_clustering": True},
                # {"threshold": 0.5, "transitive_clustering": True},
                # {"threshold": 0.5, "transitive_clustering": False},
                {"threshold": 0.75, "transitive_clustering": True},
                # {"threshold": 0.7, "transitive_clustering": False},
                # {"threshold": 0.8, "transitive_clustering": True},
                # {"threshold": 0.9, "transitive_clustering": True},
                # {"threshold": 0.7}
            ],
            save_runcase_file_path="C:\\UserData\\z0041tek\\OneDrive - Siemens AG\\Master Thesis\\Data\\"
                                   "runcases_results\\corpus_aggregation_static_tools_descriptions.json",
        )
