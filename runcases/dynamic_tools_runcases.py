import corpus_formats
import dataloaders
import techniques

from typing import Sequence
from .base import RunCase


def dynamic_tools_deduplication(ds_path: str, save_runcase_file_path: str = None) -> Sequence[RunCase]:
    dataloader = dataloaders.SefilaDataLoaderV2(
        path=ds_path,
        remove_stopwords=False,
        remove_linebreaks=True,
        remove_special_characters=False
    )
    for corpus_format in [
        corpus_formats.zap_arachni_name_description_solution,
        # corpus_formats.zap_arachni_description_solution,
        # corpus_formats.zap_arachni_description
    ]:
        # initialize technique for different embedders
        for embedder in techniques.SbertSemanticSearch.EMBEDDERS[0:1]:
            yield RunCase(
                title=f"SbertSemanticSearch {embedder}",
                dataloader=dataloader,
                corpus_format=corpus_format,
                technique=techniques.SbertSemanticSearch(embedder=embedder),
                technique_kwargs=[
                    {"threshold": 0.85, "transitive_clustering": True}
                ],
                save_runcase_file_path=save_runcase_file_path
            )
