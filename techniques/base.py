from abc import ABC, abstractmethod
from typing import Dict, Any, Sequence, Set


class BaseTechnique(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def apply(self, corpus):
        pass

    @staticmethod
    def evaluate(labels: Dict[Any, Sequence[int]], predictions: Dict[int, Sequence[int]], verbose=False) -> dict:
        # get unique predictions
        unique_predictions = set(tuple(sorted(x)) for x in predictions.values())
        # calculate accuracy
        unique_labels = set(tuple(sorted(lbl)) for lbl in labels.values())
        matched_predictions = []
        matched_labels = []
        for prediction in unique_predictions:
            for label in unique_labels:
                if prediction == label:
                    matched_predictions.append(prediction)
                    matched_labels.append(label)
                    break

        unmatched_predictions = [pred for pred in unique_predictions if pred not in matched_predictions]
        unmatched_labels = [lbl for lbl in unique_labels if lbl not in matched_labels]

        # calculate accuracy
        accuracy = len(matched_predictions) / len(unique_predictions)

        # calculate dice score
        dice_score = len(unique_labels.intersection(unique_predictions)) / len(unique_labels.union(unique_predictions))

        # calculate metrics from paper: https://www.dbs.ifi.lmu.de/~zimek/publications/ICDE2012/ICDE12_ELKI_0_5.pdf
        # P: unique_labels, Q: unique_predictions
        def get_cluster_stats(_a, _b, _c, _d):
            return {
                "precision": _a / (_a + _c),
                "recall": _a / (_a + _b),
                "f-measure": (2 * _a) / ((2 * _a) + _b + _c),
                "rand": (_a + _d) / (_a + _b + _c + _d),
                "jaccard": _a / (_a + _b + _c)
            }

        a1 = len(unique_labels.intersection(unique_predictions))
        c1 = len(unique_labels - unique_predictions)
        b1 = len(unique_predictions - unique_labels)
        d1 = 0
        cluster_stats_1 = get_cluster_stats(a1, b1, c1, d1)

        # calculate metrics from paper http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.214.7233&rep=rep1
        # &type=pdf
        # P: unique_labels, Q: unique_predictions <-- mentioned on page 367
        def get_pairs(collection: Set) -> Set:
            results = set()
            for pair in collection:
                if len(pair) == 1:
                    results.add(pair)
                    continue
                for idx1, num1 in enumerate(pair):
                    for idx2, num2 in enumerate(pair):
                        if idx1 == idx2:
                            continue
                        results.add(tuple(sorted([num1, num2])))
            return results

        all_possible_clusters = unique_labels.union(unique_predictions)  # D
        pairs_p = get_pairs(unique_labels)
        pairs_q = get_pairs(unique_predictions)
        pairs_d = get_pairs(all_possible_clusters)
        a2 = len(pairs_p.intersection(pairs_q))
        b2 = len(pairs_q - pairs_p)
        c2 = len(pairs_p - pairs_q)
        d2 = len(pairs_d - (pairs_q.intersection(pairs_p)))
        cluster_stats_2 = get_cluster_stats(a2, b2, c2, d2)

        return_val = {
            'accuracy': accuracy,
            'dice_score': dice_score,
            # 'cluster_stats_1': cluster_stats_1,
            'cluster_stats_2': cluster_stats_2
        }
        
        if verbose:
            return_val["unmatched_labels"] = unmatched_labels
            return_val["unmatched_predictions"] = unmatched_predictions

        return return_val
