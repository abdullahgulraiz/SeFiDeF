import corpus_formats
import dataloaders
import techniques
from collections import defaultdict
from typing import Sequence, Dict, Tuple, Union
from .base import RunCase


class AggregatedDataloader(dataloaders.BaseDataLoader):
    def __init__(
            self,
            unique_keys_dataloader: dataloaders.BaseDataLoader,
            unique_keys_corpus_format: corpus_formats.CorpusFormat,
            target_corpus_dataloader: dataloaders.BaseDataLoader
    ):
        # initialize dataloaders
        self.unique_keys_dataloader = unique_keys_dataloader
        self.unique_keys_corpus_format = unique_keys_corpus_format
        self.target_corpus_dataloader = target_corpus_dataloader

    def get_corpus(self, keys: Sequence[Dict[str, Union[str, Tuple[str], bool]]], separator=' - ') -> Tuple[
        Dict[int, str],  # format: {finding_id : "finding_text"}
        Dict[
            Tuple[int, str],  # format: {(collection_id, "collection_title") : [finding_ids]}
            Sequence[int]
        ]
    ]:
        # get corpus and labels for intended format
        target_corpus, target_labels = self.target_corpus_dataloader.get_corpus(
            keys=keys, separator=separator
        )
        # get corpus and labels for unique key
        unique_key_corpus, unique_key_labels = self.unique_keys_dataloader.get_corpus(
            **self.unique_keys_corpus_format.format_dict
        )
        # aggregate entries from original corpus based on unique key
        unique_key_aggregated_corpus = defaultdict(list)
        empty_str_finding_ids = []
        for finding_id, unique_key_str in unique_key_corpus.items():
            # skip if empty string and add as original
            if len(unique_key_str) == 0:
                empty_str_finding_ids.append(finding_id)
                continue
            # add to list of aggregated findings
            unique_key_aggregated_corpus[unique_key_str].append(target_corpus[finding_id])
        # convert aggregated corpus to string
        unique_key_aggregated_corpus = {
            key: separator.join(val)
            for key, val in unique_key_aggregated_corpus.items()
        }
        # form final corpus
        corpus = {key: unique_key_aggregated_corpus[val] for key, val in unique_key_corpus.items() if val != ''}
        for finding_id in empty_str_finding_ids:
            # replace if not already existing
            if finding_id not in corpus:
                corpus[finding_id] = target_corpus[finding_id]
        return dict(corpus), dict(target_labels)


def equality_comparison_static_tools(unique_ds_path: str, target_ds_path: str) -> Sequence[RunCase]:
    def _get_dataloader(ds_path: str):
        if ".json" in ds_path:
            return dataloaders.SefilaDataLoaderV2(ds_path)
        elif ".pkl" in ds_path:
            return dataloaders.PickleDataLoader(ds_path)
        else:
            raise NotImplementedError

    cve_ids_dataloader = _get_dataloader(unique_ds_path)
    descriptions_dataloader = _get_dataloader(target_ds_path)
    unique_corpus_format = corpus_formats.anchore_trivy_cve_id
    dataloader = AggregatedDataloader(cve_ids_dataloader, unique_corpus_format, descriptions_dataloader)
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
            print_report=True
        )
