import re
import json
from typing import Dict, Sequence, Tuple
from pathlib import Path
import concurrent.futures
import numpy as np
import numpy.typing as npt
from sematch.semantic.similarity import WordNetSimilarity

import utils
from techniques import BaseTechnique


class KnowledgeGraphBagOfWordsSimilarity(BaseTechnique):
    def __init__(self):
        # initialize model
        self.ontology = "Sematch WordNet"
        self.model = WordNetSimilarity()
        self.saved_word_similarities = {}
        self.saved_sentence_similarities = {}
        # load skip words from given file
        self.skip_words = {}
        self.words_to_skip_file_name = Path.cwd() / "techniques/kg_similarity" / "words_to_skip.json"
        with open(self.words_to_skip_file_name, 'r') as f:
            content = json.load(f)
            self.skip_words = content[self.ontology] if self.ontology in content else []
            self.skip_words_file_content = content  # for later saving
        # transform skip words to dictionary
        self.skip_words = {word: True for word in self.skip_words}

    def _update_skip_words_file(self):
        # transform skip words back to dictionary
        self.skip_words = [word for word, should_skip in self.skip_words.items() if should_skip]
        # skip saving to file if content is similar
        if self.skip_words_file_content[self.ontology] == self.skip_words:
            return
        # update skip words file otherwise
        with open(self.words_to_skip_file_name, 'w') as f:
            self.skip_words_file_content[self.ontology] = self.skip_words
            json.dump(self.skip_words_file_content, f)

    def _compute_words_similarity_score(self, word_1: str, word_2: str) -> float:
        try:
            score = self.model.word_similarity(word_1, word_2)
        except Exception:
            score = 0.0
        return score

    @staticmethod
    def _get_saved_similarity(
            string_1: str, string_2: str, saved_collection: Dict[Tuple[str, str], float]
    ) -> (bool, float):
        combination = (string_1, string_2)
        if combination in saved_collection:
            return True, saved_collection[combination]
        return False, None

    @staticmethod
    def _save_string_similarity(
            string_1: str, string_2: str, similarity: float, saved_collection: Dict[Tuple[str, str], float]
    ) -> None:
        for combination in [(string_1, string_2), (string_2, string_1)]:
            saved_collection[combination] = similarity

    def _compute_sentence_similarity_score(self, sentence_1: str, sentence_2: str) -> float:
        # return highest similarity if sentences are the same
        if sentence_1 == sentence_2:
            return 1.0
        # similarity is least if one of the sentences is an empty string
        if len(sentence_1) == 0 or len(sentence_2) == 0:
            return 0.0
        # remove stopwords
        sentence_1, sentence_2 = utils.remove_stopwords((sentence_1, sentence_2))
        # remove digits and special characters, since irrelevant for KG
        sentence1, sentence2 = re.sub("[^A-Za-z ]+", "", sentence_1), re.sub("[^A-Za-z ]+", "", sentence_2)
        # remove any extra whitespaces
        sentence1, sentence2 = ' '.join(sentence1.split()), ' '.join(sentence2.split())
        # convert words to list of words
        sentence1 = sentence1.split(" ")
        sentence2 = sentence2.split(" ")
        all_words = {*sentence1, *sentence2}
        # check for words not present in ontology
        for word in all_words:
            if word not in self.skip_words:
                self.skip_words[word] = self._compute_words_similarity_score(word, word) == 0
        # compute similarity
        word_similarity_count = 0
        total_similarity = 0.0
        for word1 in sentence1:
            for word2 in sentence2:
                # assign max similarity if words are same
                if word1 == word2:
                    total_similarity += 1.0
                    word_similarity_count += 1
                    continue
                # skip if words not present in ontology
                if self.skip_words[word1] or self.skip_words[word2]:
                    continue
                # check if word similarity already computed
                saved_similarity_available, similarity_score = self._get_saved_similarity(
                    string_1=word1, string_2=word2, saved_collection=self.saved_word_similarities
                )
                if not saved_similarity_available:
                    # compute word-net similarity and save for future use
                    similarity_score = self._compute_words_similarity_score(word1, word2)
                    self._save_string_similarity(
                        string_1=word1,
                        string_2=word2,
                        similarity=similarity_score,
                        saved_collection=self.saved_word_similarities
                    )
                # aggregate
                total_similarity += similarity_score
                word_similarity_count += 1
        # normalize
        if word_similarity_count > 0:
            total_similarity /= word_similarity_count
            return total_similarity
        else:
            return 0.0

    def _compute_text_matrix_similarity(self, matrix) -> npt.NDArray:
        blank_sentences_indices = matrix[0] == ""
        assert np.all(blank_sentences_indices == (matrix[1] == "")), ("Sentences for comparison should either both be "
                                                                      "blank or available.")
        results_matrix = np.full(matrix.shape[1:], -1, dtype=float)
        # create arrays of indices, first sentences, and second sentences
        non_empty_str_indices = np.argwhere(matrix[0] != "").flatten().tolist()
        sentence_1_list = matrix[0][non_empty_str_indices].tolist()
        sentence_2_list = matrix[1][non_empty_str_indices].tolist()

        def _multithread_similarity_calculation(idx: int, sentence_1: str, sentence_2: str) -> Tuple[int, float]:
            score = self._compute_sentence_similarity_score(sentence_1, sentence_2)
            return idx, score
        # get results asynchronously
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = executor.map(
                _multithread_similarity_calculation, non_empty_str_indices, sentence_1_list, sentence_2_list
            )
        for index, score in results:
            results_matrix[index] = score
        return results_matrix

    def apply(
            self, corpus: Dict[int, str], threshold: float = 0.25, transitive_clustering: bool = True
    ) -> Dict[int, Sequence[int]]:
        # convert finding Ids to array
        finding_ids = np.array(list(corpus.keys()))
        finding_texts = np.array(list(corpus.values()))
        num_entries = len(corpus)
        # create a matrix from list of finding Ids
        finding_ids_matrix = np.tile(finding_ids, (num_entries, 1))
        finding_texts_matrix = np.tile(finding_texts, (num_entries, 1))
        # create combinations of finding texts to be compared together
        finding_texts_combinations = np.array([finding_texts_matrix, np.transpose(finding_texts_matrix)])
        # positions where one of the strings is empty has 0.0 similarity
        positions_with_one_entry_empty = (
                ((finding_texts_combinations[0] == "") & (finding_texts_combinations[1] != ""))
                | ((finding_texts_combinations[1] == "") & (finding_texts_combinations[0] != ""))
        )
        # positions where strings are equal have a 1.0 similarity
        positions_with_similar_strings = finding_texts_combinations[0] == finding_texts_combinations[1]
        # similarities are mirrored across the diagonal, so we need to compute just one set of similarities
        indices_of_lower_triangle = np.tril_indices(num_entries)
        positions_of_lower_triangle = np.zeros(finding_texts_combinations.shape[1:], dtype=bool)
        positions_of_lower_triangle[indices_of_lower_triangle] = True
        positions_of_lower_triangle = np.array([positions_of_lower_triangle, positions_of_lower_triangle])
        # compute positions for which similarity needs to be computed
        positions_to_compute_similarity = ~(positions_with_one_entry_empty
                                            | positions_with_similar_strings
                                            | positions_of_lower_triangle)
        # get matrix of strings to compute for
        texts_to_compute = np.where(positions_to_compute_similarity, finding_texts_combinations, "")
        # re-shape the matrix into a vector-like form for our function to process
        texts_to_compute = texts_to_compute.reshape(texts_to_compute.shape[0], -1)
        similarities = self._compute_text_matrix_similarity(texts_to_compute)
        # re-shape back to our form. should be a lower triangular matrix containing similarities.
        similarities = similarities.reshape((num_entries, num_entries))
        # ensure spatial integrity of returned data
        assert np.all(
            (positions_to_compute_similarity[0] | positions_to_compute_similarity[1]) == (similarities >= 0)), (
            "Similarity scores values should correspond to positions where similarities should be computed."
        )
        # reflect similarity values along the diagonal
        similarities = np.triu(similarities)
        similarities = similarities + similarities.T - np.diag(np.diag(similarities))
        # add known similarities
        similarities[positions_with_one_entry_empty] = 0.0
        similarities[positions_with_similar_strings] = 1.0
        # threshold values and reduce with finding Ids
        reduced_similarities = np.where(similarities >= threshold, finding_ids_matrix, -1)
        results = {}
        # create final results
        for idx, finding_id in enumerate(finding_ids):
            results_temp = reduced_similarities[idx]
            results[finding_id] = np.delete(results_temp, results_temp == -1).tolist()
        # update skip words file
        # self._update_skip_words_file()
        # normalize clusters based on transitive property if required
        if transitive_clustering:
            results = self._transitive_clustering(results)
        return dict(results)
