import re
import json
import utils
from typing import Dict, Tuple, Sequence, Union, TypedDict, List
from collections import defaultdict
from dataloaders.base import BaseDataLoader


class SefilaDataLoader(BaseDataLoader):
    def __init__(self, path: str, remove_stopwords: bool = True) -> None:
        with open(path, 'r') as f:
            self.data = json.load(f)
        self.remove_stopwords = remove_stopwords

    def get_corpus(self, keys: Sequence[Dict[str, Union[str, Tuple[str], bool]]], separator=' - ') -> Tuple[
        Dict[int, str],  # format: {finding_id : "finding_text"}
        Dict[
            Tuple[int, str],  # format: {(collection_id, "collection_title") : [finding_ids]}
            Sequence[int]
        ]
    ]:
        # create a mapping of tools -> fields to get fields for a tool conveniently and tools -> field requirement
        fields_per_tool = dict()
        # create a mapping of tools -> field requirement to determine whether to ensure all fields for a tool
        all_fields_required_per_tool = defaultdict(bool)
        for key in keys:
            fields_per_tool[key['tool']] = key['fields']
            if 'ensure_fields' in key.values():
                all_fields_required_per_tool[key['tool']] = key['ensure_fields']
        # create empty corpus dict to store finding ID -> corpus body
        corpus = defaultdict(str)
        # create empty labels dict to store collection ID -> sequence of finding IDs
        labels = defaultdict(list)
        # start generating corpus
        for tool, fields in fields_per_tool.items():
            for collection in self.data:
                for finding in collection['findings']:
                    # skip the finding if it doesn't belong to our tool of interest
                    if tool != finding['tool']:
                        continue
                    finding_body = finding['finding']
                    # generate single corpus entry from given tool fields
                    corpus_entry = ""
                    for field in fields:
                        if field not in finding_body.keys():
                            if not all_fields_required_per_tool[tool]:
                                continue
                            raise KeyError(f"Cannot find field `{field}` for finding ID `{finding['id']}`")
                        corpus_entry += finding_body[field] + separator
                    # store corpus entry against finding ID as label
                    finding_id = int(finding['id'])
                    collection_identifier = (int(collection['id']), collection['name'])
                    # remove line breaks, tabs, and spaces and trim whitespaces
                    corpus_entry = re.sub("[ \\t\\n\\r]+", " ", corpus_entry).strip()
                    # remove special characters
                    corpus_entry = re.sub("/[\\s()-]+/gi", "", corpus_entry)
                    if self.remove_stopwords:
                        corpus_entry = utils.remove_stopwords(corpus_entry)
                    corpus[finding_id] = corpus_entry
                    labels[collection_identifier].append(finding_id)
        return dict(corpus), dict(labels)
