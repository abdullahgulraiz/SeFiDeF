import corpus_formats
import dataloaders
from collections import defaultdict
from typing import Sequence, Dict, Tuple, Union


class AggregatedDataloader(dataloaders.BaseDataLoader):
    def __init__(
        self,
        unique_keys_dataloader: dataloaders.BaseDataLoader,
        unique_keys_corpus_format: corpus_formats.CorpusFormat,
        target_corpus_dataloader: dataloaders.BaseDataLoader,
    ):
        # initialize dataloaders
        self.unique_keys_dataloader = unique_keys_dataloader
        self.unique_keys_corpus_format = unique_keys_corpus_format
        self.target_corpus_dataloader = target_corpus_dataloader

    def get_corpus(
        self, keys: Sequence[Dict[str, Union[str, Tuple[str], bool]]], separator=" - "
    ) -> Tuple[
        Dict[int, str],  # format: {finding_id : "finding_text"}
        Dict[
            Tuple[
                int, str
            ],  # format: {(collection_id, "collection_title") : [finding_ids]}
            Sequence[int],
        ],
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
            unique_key_aggregated_corpus[unique_key_str].append(
                target_corpus[finding_id]
            )
        # convert aggregated corpus to string
        unique_key_aggregated_corpus = {
            key: separator.join(val)
            for key, val in unique_key_aggregated_corpus.items()
        }
        # form corpus of values that were aggregated
        aggregated_corpus = {
            key: unique_key_aggregated_corpus[val]
            for key, val in unique_key_corpus.items()
            if val != ""
        }
        # modify target corpus to replace original entries with their corresponding aggregated entries
        for finding_id, agg_corpus_str in aggregated_corpus.items():
            target_corpus[finding_id] = agg_corpus_str
        return dict(target_corpus), dict(target_labels)
