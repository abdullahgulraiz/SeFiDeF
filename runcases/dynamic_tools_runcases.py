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
        corpus_formats.zap_arachni_name_description_solution,  # this is generally enough
        # corpus_formats.zap_arachni_description_solution,
        # corpus_formats.zap_arachni_description
    ]:
        # S-BERT - Transformer based
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

        # # Gensim - Corpus based
        # initial_corpus, _ = dataloader.get_corpus(**corpus_format.format_dict)
        # for num_topics in [  # recommended in range 200-500
        #     # 250,
        #     350,  # middle
        #     # 450
        # ]:
        #     yield RunCase(
        #         title=f"GensimLsiSimilarity Topics {num_topics}",
        #         dataloader=dataloader,
        #         corpus_format=corpus_format,
        #         technique=techniques.GensimLsiSimilarity(corpus=initial_corpus, num_topics=num_topics),
        #         technique_kwargs=[
        #             # {"threshold": 0.3, "transitive_clustering": True},
        #             # {"threshold": 0.3, "transitive_clustering": False},
        #             {"threshold": 0.5, "transitive_clustering": True},
        #             # {"threshold": 0.5, "transitive_clustering": False},
        #             # {"threshold": 0.7, "transitive_clustering": True}
        #         ],
        #         save_runcase_file_path=save_runcase_file_path
        #     )
