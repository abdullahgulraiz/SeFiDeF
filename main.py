from runcases.gensim_runcases import gensim_lsi_zap_arachni_runcases

if __name__ == '__main__':
    ds_path = '/home/abdullah/LRZ Sync+Share/TUM BMC/Master Thesis/Data/labeled-dataset-markus-new-with-tools.json'
    runcases = gensim_lsi_zap_arachni_runcases(ds_path=ds_path)
    for runcase in runcases:
        runcase.execute()
