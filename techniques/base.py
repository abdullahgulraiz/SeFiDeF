from abc import ABC, abstractmethod
from typing import Dict, Any, Sequence


class BaseTechnique(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def apply(self, corpus):
        pass

    @staticmethod
    def evaluate(labels: Dict[Any, Sequence[int]], predictions: Dict[int, Sequence[int]]) -> dict:
        # get unique predictions
        unique_predictions = []
        for prediction in predictions.values():
            if prediction not in unique_predictions:
                unique_predictions.append(sorted(prediction))
        # calculate accuracy
        unique_labels = [sorted(lbl) for lbl in labels.values()]
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

        accuracy = len(matched_predictions) / len(unique_predictions)

        return {
            'accuracy': accuracy,
            'unmatched_labels': unmatched_labels,
            'unmatched_predictions': unmatched_predictions
        }
