import json
from pprint import pprint
from datetime import datetime
from typing import Union, Sequence
from dataloaders.base import BaseDataLoader
from techniques.base import BaseTechnique
from corpus_formats.base import CorpusFormat


def print_runcase_report(runcases_data):
    corpus, labels = runcases_data["corpus"], runcases_data["labels"]
    for runcase in runcases_data["runcases"]:
        evaluation = runcase["evaluation"]
        unmatched_predictions = evaluation["unmatched_predictions"]
        unmatched_labels = evaluation["unmatched_labels"]
        for prediction in unmatched_predictions:
            print(f"\nPrediction: {prediction}\n")
            # gather related labels
            related_labels = []
            for finding_id in prediction:
                print(f"{finding_id}: {corpus[finding_id]}")
                for label in unmatched_labels:
                    if finding_id in label and label not in related_labels:
                        related_labels.append(label)
            print(f"\nRelated labels: {related_labels}")
            for label in related_labels:
                print(f"\n{label}")
                for finding_id in label:
                    print(f"{finding_id}: {corpus[finding_id]}")


class RunCase:
    def __init__(
        self,
        title: str,
        dataloader: BaseDataLoader,
        corpus_format: CorpusFormat,
        technique: BaseTechnique,
        technique_kwargs: Union[dict, Sequence[dict]] = None,
        save_runcase_file_path: str = None,
    ):
        self.title = title
        self.dataloader = dataloader
        self.corpus_format = corpus_format
        self.corpus, self.labels = self.dataloader.get_corpus(
            **corpus_format.format_dict
        )
        # ensure that equal findingIds are present in both corpus and labels
        corpus_finding_ids = set(findingId for findingId in self.corpus.keys())
        label_finding_ids = set(
            findingId for lbl in self.labels.values() for findingId in lbl
        )
        assert (
            corpus_finding_ids == label_finding_ids
        ), "Finding IDs should be equal in both corpus and labels."
        self.technique = technique
        self.technique_kwargs = technique_kwargs
        self.save_runcase_file_path = save_runcase_file_path

    def execute(
        self,
        print_report: bool = True,
        print_evaluation_fields: Union[Sequence, bool] = True,
        **execution_kwargs,
    ):
        # add empty kwargs if absent
        if not self.technique_kwargs:
            self.technique_kwargs = [{}]
        # convert normal kwargs to sequence
        if not isinstance(self.technique_kwargs, Sequence):
            self.technique_kwargs = [self.technique_kwargs]
        # save runcase data
        runcases_data = {
            "corpus": self.corpus,
            "labels": {str(key): val for key, val in self.labels.items()},
            "runcases": [],
        }
        # apply technique on corpus and evaluate results
        for runcase_idx, technique_kwargs in enumerate(self.technique_kwargs):
            runcase_title = (
                f"==== \n"
                f"RunCase: `{self.title}`, "
                f"Corpus: {self.corpus_format.name}, "
                f"Params: `{technique_kwargs}` "
                f"\n===="
            )
            results = self.technique.apply(self.corpus, **technique_kwargs)
            evaluation = self.technique.evaluate(self.labels, results)
            # print evaluation results
            if print_evaluation_fields:
                print(runcase_title)
                if isinstance(print_evaluation_fields, Sequence):
                    pprint(
                        {field: evaluation[field] for field in print_evaluation_fields},
                        compact=True,
                    )
                else:
                    pprint(evaluation, compact=True)
                # print for excel
                print(
                    evaluation["accuracy"]["predictions"],
                    "\t",
                    evaluation["accuracy"]["labels"],
                    "\t",
                    evaluation["accuracy"]["average"],
                    "\t",
                    evaluation["f-measure"],
                    "\t",
                    evaluation["precision"],
                    "\t",
                    evaluation["recall"],
                )
            # store runcases data
            runcases_data["runcases"].append(
                {"title": runcase_title, "results": results, "evaluation": evaluation}
            )
            # save runcase file for evaluation in SeFiLa if path provided
            # TODO: modify SeFiLa to save runcases
            if self.save_runcase_file_path:
                current_timestamp = str(datetime.now()).replace(":", ".")
                filename = f"{self.save_runcase_file_path}_{runcase_idx}_{current_timestamp}.json"
                with open(filename, "w") as f:
                    json.dump(runcases_data, f)
                print(f"RunCase {runcase_idx} results saved to: {filename}")
        # print detailed runcase report if required
        if print_report:
            print("----\nRunCase Report\n----")
            print_runcase_report(runcases_data)
