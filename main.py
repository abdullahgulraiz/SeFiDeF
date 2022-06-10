import os
from pathlib import Path

import runcases


if __name__ == '__main__':
    base_path = os.environ.get("DATA_BASE_PATH")
    if not base_path:
        raise RuntimeError("Please define the base path for data files as the environment variable DATA_BASE_PATH.")
    base_path = Path(base_path)
    sefila_dynamic_ds_path = base_path / "labeled-dataset-markus-new-with-tools.json"
    sefila_static_ds_path = base_path / "labeled-dataset-static-tools.json"
    pkl_static_ds_path = base_path / "sefidef-corpus-static-tools-scraped.pkl"
    anchore_trivy_descriptions_pkl = base_path / "anchore_trivy_descriptions_scraped.pkl"
    pkl_static_ds_path_2 = base_path / "anchore_trivy_cve_id_package_path_url_scraped.pkl"
    multiple_static_tools_ds = base_path / "static-tools-ds-all.json"
    run_cases = [
        # *runcases.gensim_lsi_zap_arachni_runcases(ds_path=sefila_dynamic_ds_path),
        # *sbert_zap_arachni_runcases(ds_path=sefila_ds_path)
        # *runcases.gensim_lsi_trivy_anchore_runcases_1(ds_path=pkl_static_ds_path),
        # *runcases.gensim_lsi_trivy_anchore_runcases_2(ds_path=pkl_static_ds_path_2),
        # *runcases.sbert_trivy_anchore_runcases_1(ds_path=pkl_static_ds_path),
        # *runcases.sbert_trivy_anchore_runcases_2(ds_path=pkl_static_ds_path_2),
        # *runcases.equality_comparison_static_tools(
        #     unique_ds_path=str(sefila_static_ds_path),
        #     target_ds_path=str(anchore_trivy_descriptions_pkl)
        # ),
        # *runcases.sbert_multiple_static_tools_descriptions(ds_path=str(multiple_static_tools_ds)),
        *runcases.corpus_aggregation_static_tools_descriptions(
            unique_ds_path=str(multiple_static_tools_ds),
            target_ds_path=str(multiple_static_tools_ds)
        )
    ]
    for run_case in run_cases:
        run_case.execute(print_report=True)
