from runcases.sbert_runcases import sbert_zap_arachni_runcases
from runcases.gensim_runcases import gensim_lsi_zap_arachni_runcases
from dataloaders.sefila_dataloader import SefilaDataLoader
from techniques.gensim_lsi_similarity import GensimLsiSimilarity
from pprint import pprint

if __name__ == '__main__':
    ds_path = '/home/abdullah/LRZ Sync+Share/TUM BMC/Master Thesis/Data/labeled-dataset-markus-new-with-tools.json'

    # runcases = sbert_zap_arachni_runcases(ds_path='/home/abdullah/LRZ Sync+Share/TUM BMC/Master Thesis/Data/labeled-'
    #                                               'dataset-markus-new-with-tools.json')

    runcases = gensim_lsi_zap_arachni_runcases(ds_path=ds_path)
    for runcase in runcases:
        runcase.execute()
    # corpus_format = {'keys': [{'tool': 'arachni',
    #                            'fields': (
    #                                'name',
    #                                'description',
    #                                'remedy_guidance'
    #                            ),
    #                            'ensure_fields': False},
    #                           {'tool': 'zap',
    #                            'fields': (
    #                                'name',
    #                                'desc',
    #                                'solution'
    #                            ),
    #                            'ensure_fields': False}],
    #                  'separator': " "}
    # dataloader = SefilaDataLoader('/home/abdullah/LRZ Sync+Share/TUM BMC/Master Thesis/Data/labeled-dataset-markus'
    #                               '-new-with-tools.json')
    # corpus, labels = dataloader.get_corpus(**corpus_format)
    # for threshold in [0.5, 0.2]:
    #     technique = GensimLsiSimilarity(corpus, threshold=threshold)
    #     results = technique.apply(corpus)
    #     evaluation = technique.evaluate(labels, results)
    #     evaluation = {'threshold': threshold, **evaluation}
    #     pprint(evaluation)
