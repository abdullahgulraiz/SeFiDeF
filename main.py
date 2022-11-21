from pathlib import Path
import multiprocess as mp

import runcases


# function to call execute method on a run case
def execute_runcase(run_case: runcases.RunCase):
    run_case.execute(
        print_report=False,
        print_evaluation_fields=("accuracy", "f-measure", "precision", "recall"),
    )


if __name__ == "__main__":
    # configure directories
    working_directory = Path().resolve()
    base_path = working_directory / "datasets"
    results_path = working_directory / "results"
    # create results directory if it doesn't exist
    results_path.mkdir(parents=True, exist_ok=True)
    # dataset paths
    static_tools_ds = base_path / "static-tools-ds.json"
    dynamic_tools_ds = base_path / "dynamic-tools-ds.json"
    # result paths
    dynamic_tools_results = results_path / "dynamic_tools_deduplication"
    static_tools_results = results_path / "static_tools_deduplication"
    run_cases = [
        # --- other experiments ---
        # *runcases.sbert_multiple_static_tools_descriptions(ds_path=str(static_tools_ds)),
        # *runcases.corpus_aggregation_static_tools_descriptions(
        #     unique_ds_path=str(static_tools_ds),
        #     target_ds_path=str(static_tools_ds)
        # ),
        # --- dynamic tools findings deduplication ---
        # *runcases.dynamic_tools_deduplication(
        #     ds_path=str(dynamic_tools_ds),
        #     save_runcase_file_path=str(dynamic_tools_results)
        # ),
        # --- static tools findings deduplication ---
        *runcases.static_tools_deduplication(
            ds_path=str(static_tools_ds),
            save_runcase_file_path=str(static_tools_results),
        )
    ]
    # create a pool of process workers
    pool = mp.Pool(mp.cpu_count())
    # call function for each run case separately
    pool.map(execute_runcase, run_cases)
    pool.close()
    # wait for all workers to complete
    pool.join()
