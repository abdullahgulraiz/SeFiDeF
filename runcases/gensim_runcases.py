from typing import Sequence
from dataloaders.sefila_dataloader import SefilaDataLoader
from techniques.gensim_lsi_similarity import GensimLsiSimilarity
from runcases.base import RunCase
from corpus_formats import zap_arachni


def gensim_lsi_zap_arachni_runcases(ds_path: str) -> Sequence[RunCase]:
    dataloader = SefilaDataLoader(ds_path)
    corpus_format = zap_arachni.name_description_solution_1
    initial_corpus, _ = dataloader.get_corpus(**corpus_format.format_dict)
    for num_topics in [100, 200, 300]:
        yield RunCase(
            title=f"GensimLsiSimilarity Topics {num_topics}",
            dataloader=dataloader,
            corpus_format=corpus_format,
            technique=GensimLsiSimilarity(corpus=initial_corpus, num_topics=num_topics),
            technique_kwargs=[{"threshold": 0.2}, {"threshold": 0.5}]
        )
