from abc import ABC, abstractmethod
from itertools import combinations
from typing import Dict, Any, Sequence


class BaseTechnique(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def apply(self, corpus):
        pass

    @staticmethod
    def _transitive_clustering(results):
        for finding_id_1, finding_cluster in results.items():
            # convert finding cluster to set
            finding_cluster_final = set(finding_cluster)
            for finding_id_2 in finding_cluster:
                # skip if same finding
                if finding_id_1 == finding_id_2:
                    continue
                # add all related findings to the formed cluster
                finding_cluster_final.update(results[finding_id_2])
            # add to original results
            results[finding_id_1] = list(finding_cluster_final)
        return results

    @staticmethod
    def evaluate(
            labels: Dict[Any, Sequence[int]],
            predictions: Dict[int, Sequence[int]],
            round_digits: int = 3
    ) -> dict:
        # establish unique predictions and labels
        unique_predictions = set(tuple(sorted(pred)) for pred in predictions.values() if len(pred) > 0)
        unique_labels = set(tuple(sorted(lbl)) for lbl in labels.values() if len(lbl) > 0)
        # ensure finding Ids are equal in both predictions and labels
        unique_predictions_finding_ids = {finding_id for cluster in unique_predictions for finding_id in cluster}
        unique_labels_finding_ids = {finding_id for cluster in unique_labels for finding_id in cluster}
        assert unique_predictions_finding_ids == unique_labels_finding_ids, ("Finding IDs should be equal in both "
                                                                             "corpus and labels.")
        # compute matched and unmatched predictions
        matched_predictions = matched_labels = unique_predictions.intersection(unique_labels)
        unmatched_predictions = unique_predictions.symmetric_difference(matched_predictions)
        unmatched_labels = unique_labels.symmetric_difference(matched_labels)
        # calculate accuracies
        prediction_accuracy = len(matched_predictions) / len(unique_predictions)
        label_accuracy = len(matched_labels) / len(unique_labels)
        total_accuracy = (prediction_accuracy + label_accuracy) / 2
        # calculate clustering statistics
        """
        Ref: https://stackoverflow.com/questions/12725263/computing-f-measure-for-clustering
        For our case:
            D: unique_predictions_finding_ids
            P: unique_predictions
            Q: unique_labels  
        """
        pairs_d = set(combinations(unique_predictions_finding_ids, 2))
        pairs_p = set(pair for cluster in unique_predictions for pair in combinations(cluster, 2))
        pairs_q = set(pair for cluster in unique_labels for pair in combinations(cluster, 2))
        a = len(pairs_p.intersection(pairs_q))
        b = len(pairs_p - pairs_q)
        c = len(pairs_q - pairs_p)
        precision = a / (a + c)
        recall = a / (a + b)
        f_measure = (2 * a) / ((2 * a) + b + c)
        return {
            'accuracy': {
                'predictions': round(prediction_accuracy, round_digits),
                'labels': round(label_accuracy, round_digits),
                'average': round(total_accuracy, round_digits)
            },
            "f-measure": f_measure,
            "precision": precision,
            "recall": recall,
            "unmatched_labels": list(unmatched_labels),
            "unmatched_predictions": list(unmatched_predictions),
            "matched_predictions": list(matched_predictions)
        }
