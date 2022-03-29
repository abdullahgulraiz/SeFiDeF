import corpus_formats
import dataloaders
import techniques
from typing import Sequence
from .base import RunCase


def gensim_lsi_zap_arachni_runcases(ds_path: str) -> Sequence[RunCase]:
    dataloader = dataloaders.SefilaDataLoaderV1(ds_path)
    corpus_format = corpus_formats.zap_arachni.name_description_solution_1
    initial_corpus, _ = dataloader.get_corpus(**corpus_format.format_dict)
    for num_topics in [100, 200, 300]:
        yield RunCase(
            title=f"GensimLsiSimilarity Topics {num_topics}",
            dataloader=dataloader,
            corpus_format=corpus_format,
            technique=techniques.GensimLsiSimilarity(corpus=initial_corpus, num_topics=num_topics),
            technique_kwargs=[{"threshold": 0.2}, {"threshold": 0.5}]
        )


def gensim_lsi_trivy_anchore_runcases(ds_path: str) -> Sequence[RunCase]:
    # dataloader = SefilaDataLoaderV2(ds_path)
    dataloader = dataloaders.PickleDataLoader(ds_path)
    corpus_format = corpus_formats.static_tools.anchore_trivy_name_title_description
    initial_corpus, _ = dataloader.get_corpus(**corpus_format.format_dict)
    for num_topics in [300]:
        yield RunCase(
            title=f"GensimLsiSimilarity Topics {num_topics}",
            dataloader=dataloader,
            corpus_format=corpus_format,
            technique=techniques.GensimLsiSimilarity(corpus=initial_corpus, num_topics=num_topics),
            technique_kwargs=[{"threshold": 0.2}, {"threshold": 0.5}]
        )
