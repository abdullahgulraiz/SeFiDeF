import runcases


if __name__ == '__main__':
    sefila_ds_path = "/home/abdullah/LRZ Sync+Share/TUM BMC/Master Thesis/Data/labeled-dataset-markus-new-with-tools.json"
    pkl_ds_path = "/home/abdullah/LRZ Sync+Share/TUM BMC/Master Thesis/Data/sefidef-corpus-static-tools-scraped.pkl"
    run_cases = [
        # *gensim_lsi_zap_arachni_runcases(ds_path=sefila_ds_path),
        # *sbert_zap_arachni_runcases(ds_path=sefila_ds_path)
        # *runcases.gensim_lsi_trivy_anchore_runcases(ds_path=pkl_ds_path)
        *runcases.sbert_trivy_anchore_runcases(ds_path=pkl_ds_path)
    ]
    for run_case in run_cases:
        run_case.execute()

