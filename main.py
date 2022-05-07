import runcases


if __name__ == '__main__':
    sefila_dynamic_ds_path = "/home/abdullah/LRZ Sync+Share/TUM BMC/Master Thesis/Data/labeled-dataset-markus-new-with-tools.json"
    sefila_static_ds_path = "/home/abdullah/LRZ Sync+Share/TUM BMC/Master Thesis/Data/labeled-dataset-static-tools.json"
    pkl_static_ds_path = "/home/abdullah/LRZ Sync+Share/TUM BMC/Master Thesis/Data/sefidef-corpus-static-tools-scraped.pkl"
    anchore_trivy_descriptions_pkl = "/home/abdullah/LRZ Sync+Share/TUM BMC/Master Thesis/Data/anchore_trivy_descriptions_scraped.pkl"
    pkl_static_ds_path_2 = "/home/abdullah/LRZ Sync+Share/TUM BMC/Master Thesis/Data/anchore_trivy_cve_id_package_path_url_scraped.pkl"
    run_cases = [
        # *runcases.gensim_lsi_zap_arachni_runcases(ds_path=sefila_dynamic_ds_path),
        # *sbert_zap_arachni_runcases(ds_path=sefila_ds_path)
        # *runcases.gensim_lsi_trivy_anchore_runcases_1(ds_path=pkl_static_ds_path),
        # *runcases.gensim_lsi_trivy_anchore_runcases_2(ds_path=pkl_static_ds_path_2),
        # *runcases.sbert_trivy_anchore_runcases_1(ds_path=pkl_static_ds_path),
        # *runcases.sbert_trivy_anchore_runcases_2(ds_path=pkl_static_ds_path_2),
        *runcases.equality_comparison_static_tools(
            unique_ds_path=sefila_static_ds_path,
            target_ds_path=anchore_trivy_descriptions_pkl
        )
    ]
    for run_case in run_cases:
        run_case.execute(verbose=False)
