import re
import json
import utils
from typing import Dict, Tuple, Sequence, Union, TypedDict, List
from collections import defaultdict
from dataloaders.base import BaseDataLoader


class SefilaDataLoaderV1(BaseDataLoader):
    def __init__(
            self,
            path: str,
            remove_stopwords: bool = True,
            remove_linebreaks: bool = True,
            remove_special_characters: bool = True,
            to_lowercase: bool = True
    ) -> None:
        with open(path, 'r') as f:
            self.data = json.load(f)
        self.remove_stopwords = remove_stopwords
        self.remove_linebreaks = remove_linebreaks
        self.remove_special_characters = remove_special_characters
        self.to_lowercase = to_lowercase

    def _get_collections(self):
        return self.data

    @staticmethod
    def _get_tool_for_finding(finding):
        return finding['tool']

    def get_corpus(self, keys: Sequence[Dict[str, Union[str, Tuple[str], bool]]], separator=' - ') -> Tuple[
        Dict[int, str],  # format: {finding_id : "finding_text"}
        Dict[
            Tuple[int, str],  # format: {(collection_id, "collection_title") : [finding_ids]}
            Sequence[int]
        ]
    ]:
        # create a mapping of tools -> fields to get fields for a tool conveniently and tools -> field requirement
        fields_per_tool = dict()
        processing_functions_per_tool = defaultdict(dict)
        # create a mapping of tools -> field requirement to determine whether to ensure all fields for a tool
        all_fields_required_per_tool = defaultdict(bool)
        # create a mapping of field
        for key in keys:
            tool_name = key['tool']
            fields_per_tool[tool_name] = key['fields']
            if 'ensure_fields' in key.keys():
                all_fields_required_per_tool[tool_name] = key['ensure_fields']
            if 'processing_functions' in key.keys():
                processing_functions_per_tool[tool_name] = key['processing_functions']
        # create empty corpus dict to store finding ID -> corpus body
        corpus = defaultdict(str)
        # create empty labels dict to store collection ID -> sequence of finding IDs
        labels = defaultdict(list)
        # start generating corpus
        for tool, fields in fields_per_tool.items():
            for collection in self._get_collections():
                for finding in collection['findings']:
                    # skip the finding if it doesn't belong to our tool of interest
                    if tool != self._get_tool_for_finding(finding):
                        continue
                    finding_body = finding['finding']
                    # generate single corpus entry from given tool fields
                    corpus_entry = []
                    for field in fields:
                        if field not in finding_body.keys():
                            if not all_fields_required_per_tool[tool]:
                                continue
                            raise KeyError(f"Cannot find field `{field}` for finding ID `{finding['id']}`")
                        # get field value
                        field_value = finding_body[field]
                        # process field value through provided function if applicable
                        if tool in processing_functions_per_tool.keys():
                            processing_functions = processing_functions_per_tool[tool]
                            if field in processing_functions.keys():
                                field_value = processing_functions[field](field_value)
                        corpus_entry.append(field_value)
                    # join corpus entry with seperator
                    corpus_entry = separator.join(corpus_entry)
                    # store corpus entry against finding ID as label
                    finding_id = int(finding['id'])
                    collection_identifier = (int(collection['id']), collection['name'])
                    # remove line breaks, tabs, and spaces and trim whitespaces
                    if self.remove_linebreaks:
                        corpus_entry = re.sub("[ \\t\\n\\r]+", " ", corpus_entry).strip()
                    # remove special characters
                    if self.remove_special_characters:
                        corpus_entry = re.sub("/[\\s()-]+/gi", "", corpus_entry)
                    if self.remove_stopwords:
                        corpus_entry = utils.remove_stopwords(corpus_entry)
                    if self.to_lowercase:
                        corpus_entry = corpus_entry.lower()
                    corpus[finding_id] = corpus_entry
                    labels[collection_identifier].append(finding_id)
        return dict(corpus), dict(labels)


class SefilaDataLoaderV2(SefilaDataLoaderV1):
    def __init__(
            self,
            path: str,
            remove_stopwords: bool = True,
            remove_linebreaks: bool = True,
            remove_special_characters: bool = True,
            to_lowercase: bool = True
    ) -> None:
        super().__init__(path, remove_stopwords, remove_linebreaks, remove_special_characters, to_lowercase)
        # create finding id -> tool name mapping for efficient retrieval later
        self.finding_id_tool_name_mapping = {}
        for metadata in self.data["metadata"]:
            tool = metadata["tool"]
            for i in range(metadata["startIndex"], metadata["endIndex"] + 1, 1):
                self.finding_id_tool_name_mapping[i] = tool

    def _get_collections(self):
        return self.data["collections"]

    def _get_tool_for_finding(self, finding):
        return self.finding_id_tool_name_mapping[
            int(finding["id"])
        ]
