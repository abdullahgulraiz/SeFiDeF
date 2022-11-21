import utils
from copy import deepcopy
from collections import defaultdict
from typing import Dict, Sequence
from gensim import corpora, models, similarities
from techniques.base import BaseTechnique


class GensimLsiSimilarity(BaseTechnique):
    def __init__(
        self, corpus: Dict[int, str], tf_idf: bool = True, num_topics: int = 300
    ):
        self.param_corpus = corpus
        self.training_corpus = list(corpus.values())
        # remove common words and tokenize
        texts = utils.remove_stopwords(self.training_corpus, tokenize=True)
        # remove words that appear only once
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        texts = [[token for token in text if frequency[token] > 1] for text in texts]
        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.tf_idf = None
        if tf_idf:
            tfidf = models.TfidfModel(corpus)
            corpus = [tfidf[text] for text in corpus]
            self.tf_idf = tfidf
        self.gensim_model = models.LsiModel(
            corpus, id2word=self.dictionary, num_topics=num_topics
        )
        # transform corpus to LSI space and index it
        self.corpus_index = similarities.MatrixSimilarity(self.gensim_model[corpus])

    def apply(
        self,
        corpus: Dict[int, str],
        threshold: float = 0.5,
        transitive_clustering: bool = True,
    ) -> Dict[int, Sequence[int]]:
        # create string -> Id mapping
        string_id_mapping = defaultdict(list)
        for _id, corpus_string in corpus.items():
            string_id_mapping[corpus_string].append(_id)
        # sort result into corpus id -> sequence of related findings that pass our threshold
        results = defaultdict(list)
        empty_query_ids = []
        for query_id, query in corpus.items():
            # manually handle empty queries, since no bag of words can be generated for them
            if len(query) == 0:
                empty_query_ids.append(query_id)
                continue
            vec_bow = self.dictionary.doc2bow(query.lower().split())
            if self.tf_idf:
                vec_bow = self.tf_idf[vec_bow]
            vec_model = self.gensim_model[vec_bow]
            sims = self.corpus_index[vec_model]
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            string_id_mapping_copy = deepcopy(string_id_mapping)
            results[query_id] = []
            for doc_position, doc_score in sims:
                if float(doc_score) < threshold:
                    continue
                doc_text = self.training_corpus[doc_position]
                doc_id = string_id_mapping_copy[doc_text].pop()
                if doc_id not in results[query_id]:
                    results[query_id].append(doc_id)
            results[query_id].sort()

        # add results for empty queries in final results
        empty_query_ids = sorted(empty_query_ids)
        for query_id in empty_query_ids:
            results[query_id] = empty_query_ids

        # normalize clusters based on transitive property if required
        if transitive_clustering:
            results = self._transitive_clustering(results)
        return dict(results)
