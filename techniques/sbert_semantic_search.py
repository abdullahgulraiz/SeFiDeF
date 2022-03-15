from copy import deepcopy
from typing import Dict, Sequence
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from techniques.base import BaseTechnique


class SbertSemanticSearch(BaseTechnique):
    EMBEDDERS = [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2',
        'multi-qa-MiniLM-L6-cos-v1',
        'multi-qa-mpnet-base-dot-v1'
    ]

    def __init__(self, embedder: str) -> None:
        self.embedder = SentenceTransformer(embedder)

    def apply(self, corpus: Dict[int, str], threshold: float = 0.2) -> Dict[int, Sequence[int]]:
        # ensure we have a threshold variable
        corpus_strings = list(corpus.values())
        corpus_embeddings = self.embedder.encode(corpus_strings, convert_to_tensor=True)
        # get query results from technique
        query_results = util.semantic_search(corpus_embeddings, corpus_embeddings, top_k=len(corpus_strings))
        # create string -> Id mapping
        string_id_mapping = defaultdict(list)
        for _id, corpus_string in corpus.items():
            string_id_mapping[corpus_string].append(_id)
        # sort result into corpus id -> sequence of related findings that pass our threshold
        results = defaultdict(list)

        # function to get string id for a search result
        def get_string_id_from_search_result(_string_id_mapping: defaultdict, _search_result: dict) -> int:
            # get string of the result
            _corpus_string = corpus_strings[_search_result['corpus_id']]
            # get corresponding id from the original corpus
            return _string_id_mapping[_corpus_string].pop()

        # create a copy of string id mapping for the search strings (we don't want the same string Id's to repeat for
        # same string)
        string_id_mapping_search_string = deepcopy(string_id_mapping)
        for query_result in query_results:
            # create a copy of string id mapping, since it will get modified for every query result
            string_id_mapping_other_results = deepcopy(string_id_mapping)
            # query results are based on descending similarity score, so first result would be the search term itself
            search_string_id = get_string_id_from_search_result(string_id_mapping_search_string, query_result[0])
            results[search_string_id] = []  # initiate with empty list
            # iterate over other results and see if
            for other_result in query_result:
                # only include the result if it crosses threshold (if applicable)
                if float(other_result['score']) < threshold:
                    continue
                # get corresponding id from the original corpus
                corpus_string_id = get_string_id_from_search_result(string_id_mapping_other_results, other_result)
                # add id to results if it doesn't already exist
                if corpus_string_id not in results[search_string_id]:
                    # save result in hits for query string
                    results[search_string_id].append(corpus_string_id)
            # sort the list for easier processing in future
            results[search_string_id].sort()

        return dict(results)
