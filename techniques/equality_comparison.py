from collections import defaultdict
from typing import Dict, Sequence
from techniques import BaseTechnique


class EqualityComparison(BaseTechnique):
    def __init__(self, skip_blank=False):
        self.skip_blank = skip_blank

    def apply(self, corpus: Dict[int, str]) -> Dict[int, Sequence[int]]:
        results = defaultdict(list)
        for finding_id_main, finding_text_main in corpus.items():
            # skip comparison if option enabled and text is blank
            if self.skip_blank and not finding_text_main:
                results[finding_id_main].append(finding_id_main)
                continue
            for finding_id_sub, finding_text_sub in corpus.items():
                if finding_text_main == finding_text_sub:
                    results[finding_id_main].append(finding_id_sub)
            results[finding_id_main].sort()
        return dict(results)
