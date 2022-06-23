import os
from pathlib import Path
import multiprocess as mp

import runcases


# function to call execute method on a run case
def execute_runcase(run_case: runcases.RunCase):
    run_case.execute(print_report=False, print_evaluation_fields=('accuracy',))


if __name__ == '__main__':
    base_path = os.environ.get("DATA_BASE_PATH")
    results_path = os.environ.get("RESULTS_SAVE_PATH")
    if not (base_path or results_path):
        raise RuntimeError("Please define the base path for data files as the environment variable DATA_BASE_PATH.")
    base_path = Path(base_path)
    results_path = Path(results_path)
    # dataset paths
    sefila_dynamic_ds_path = base_path / "labeled-dataset-markus-new-with-tools.json"
    sefila_static_ds_path = base_path / "labeled-dataset-static-tools.json"
    pkl_static_ds_path = base_path / "sefidef-corpus-static-tools-scraped.pkl"
    anchore_trivy_descriptions_pkl = base_path / "anchore_trivy_descriptions_scraped.pkl"
    pkl_static_ds_path_2 = base_path / "anchore_trivy_cve_id_package_path_url_scraped.pkl"
    multiple_static_tools_ds = base_path / "static-tools-ds-all.json"
    dynamic_tools_ds = base_path / "dynamic_dataset_formatted.json"
    # result paths
    dynamic_tools_results = results_path / "dynamic_tools_deduplication"
    static_tools_results = results_path / "static_tools_deduplication"
    run_cases = [
        # *runcases.equality_comparison_static_tools(
        #     unique_ds_path=str(sefila_static_ds_path),
        #     target_ds_path=str(anchore_trivy_descriptions_pkl)
        # ),
        # *runcases.sbert_multiple_static_tools_descriptions(ds_path=str(multiple_static_tools_ds)),
        # *runcases.corpus_aggregation_static_tools_descriptions(
        #     unique_ds_path=str(multiple_static_tools_ds),
        #     target_ds_path=str(multiple_static_tools_ds)
        # ),
        # *runcases.dynamic_tools_deduplication(
        #     ds_path=str(dynamic_tools_ds),
        #     save_runcase_file_path=str(dynamic_tools_results)
        # ),
        *runcases.static_tools_deduplication(
            ds_path=str(multiple_static_tools_ds),
            save_runcase_file_path=None  # str(static_tools_results)
        )
    ]
    # create a pool of process workers
    pool = mp.Pool(mp.cpu_count())
    # call function for each run case separately
    pool.map(execute_runcase, run_cases)
    pool.close()
    # wait for all workers to complete
    pool.join()
