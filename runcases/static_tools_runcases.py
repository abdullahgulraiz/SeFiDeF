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
    dataloader = dataloaders.AggregatedDataloader(
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


def static_tools_deduplication(ds_path: str, save_runcase_file_path: str = None) -> Sequence[RunCase]:
    # corpus formats
    cve_ids_corpus_format = corpus_formats.multiple_static_tools_ds_cve_ids
    descriptions_corpus_format = corpus_formats.multiple_static_tools_ds_descriptions
    # dataloaders
    cve_ids_dataloader = descriptions_dataloader = dataloaders.SefilaDataLoaderV2(
        path=ds_path, remove_stopwords=False, remove_linebreaks=True, remove_special_characters=False
    )
    aggregated_descriptions_dataloader = dataloaders.AggregatedDataloader(
        unique_keys_dataloader=cve_ids_dataloader,
        unique_keys_corpus_format=cve_ids_corpus_format,
        target_corpus_dataloader=descriptions_dataloader
    )
    _dataloaders = {
        'Descriptions': descriptions_dataloader,
        'Aggregated Descriptions': aggregated_descriptions_dataloader
    }
    # corpus (for corpus-based methods' initialization)
    descriptions_corpus, _ = descriptions_dataloader.get_corpus(**descriptions_corpus_format.format_dict)
    aggregated_descriptions_corpus, _ = aggregated_descriptions_dataloader.get_corpus(
        **descriptions_corpus_format.format_dict
    )
    _corpus = {
        'Descriptions': descriptions_corpus,
        'Aggregated Descriptions': aggregated_descriptions_corpus
    }
    # techniques
    sbert_semantic_search = techniques.SbertSemanticSearch(embedder=techniques.SbertSemanticSearch.EMBEDDERS[0])
    kg_similarity = techniques.KnowledgeGraphBagOfWordsSimilarityV2()
    _techniques_kwargs = {
        "SbertSemanticSearch": [
            {"threshold": 0.1},
            {"threshold": 0.2},
            {"threshold": 0.3},
            {"threshold": 0.4},
            {"threshold": 0.5},
            {"threshold": 0.6},
            {"threshold": 0.7},
            {"threshold": 0.8},
            {"threshold": 0.9},
            {"threshold": 0.95},  # max accuracy
        ],
        "GensimLsiSimilarity": [
            {"threshold": 0.1},
            {"threshold": 0.2},
            {"threshold": 0.3},
            {"threshold": 0.4},
            {"threshold": 0.5},
            {"threshold": 0.6},
            {"threshold": 0.7},
            {"threshold": 0.8},
            {"threshold": 0.9},  # maximum accuracy
            {"threshold": 0.95},
        ],
        "KgSimilarity": [
            {"threshold": 0.1},
            {"threshold": 0.2},
            {"threshold": 0.3},
            {"threshold": 0.4},
            {"threshold": 0.5},
            {"threshold": 0.6},
            {"threshold": 0.7},
            {"threshold": 0.8},
            {"threshold": 0.9},
            {"threshold": 0.95},
        ]
    }

    def _get_technique(_technique_name: str, _dataloader_name: str = None) -> techniques.BaseTechnique:
        if _technique_name == "SbertSemanticSearch":
            return sbert_semantic_search
        elif _technique_name == "GensimLsiSimilarity":
            return techniques.GensimLsiSimilarity(corpus=_corpus[_dataloader_name], num_topics=350)
        elif _technique_name == "KgSimilarity":
            return kg_similarity

    for dataloader_name in [
        # "Descriptions",
        "Aggregated Descriptions"
    ]:
        for technique_name in [
            # "SbertSemanticSearch",
            # "GensimLsiSimilarity",
            "KgSimilarity"
        ]:
            dataloader = _dataloaders[dataloader_name]
            yield RunCase(
                title=f"{dataloader_name}, {technique_name}",
                dataloader=dataloader,
                corpus_format=descriptions_corpus_format,
                technique=_get_technique(technique_name, dataloader_name),
                technique_kwargs=_techniques_kwargs[technique_name],
                save_runcase_file_path=save_runcase_file_path
            )
