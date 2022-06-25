import re
import json
from collections import defaultdict
from typing import Dict, Sequence, Tuple
from pathlib import Path
from sematch.semantic.similarity import WordNetSimilarity

from techniques import BaseTechnique
import utils


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
        return self.model.word_similarity(word_1, word_2)

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
        all_words = [*sentence1, *sentence2]
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
        total_similarity /= word_similarity_count
        return total_similarity

    def apply(self, corpus: Dict[int, str], threshold: float = 0.25) -> Dict[int, Sequence[int]]:
        results = defaultdict(list)
        for finding_id_main, finding_text_main in corpus.items():
            # finding is similar to itself
            results[finding_id_main].append(finding_id_main)
            # get all other similar findings
            for finding_id_sec, finding_text_sec in corpus.items():
                saved_similarity_available, similarity_score = self._get_saved_similarity(
                    string_1=finding_text_main,
                    string_2=finding_text_sec,
                    saved_collection=self.saved_sentence_similarities
                )
                if not saved_similarity_available:
                    similarity_score = self._compute_sentence_similarity_score(finding_text_main, finding_text_sec)
                    self._save_string_similarity(
                        string_1=finding_text_main,
                        string_2=finding_text_sec,
                        similarity=similarity_score,
                        saved_collection=self.saved_sentence_similarities
                    )
                if similarity_score >= threshold:
                    if finding_id_sec not in results[finding_id_main]:
                        results[finding_id_main].append(finding_id_sec)
            results[finding_id_main].sort()
        # update skip words file
        self._update_skip_words_file()
        return dict(results)
