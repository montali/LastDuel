import jsonlines
from openml import OpenMLTask, OpenMLRun
import openml
from typing import List, Dict
from p_tqdm import p_map
import pandas as pd


def download_runs(flow: Dict) -> List[Dict]:
    flow_runs = []
    tasks = {}
    datasets = {}
    for run_id in list(openml.runs.list_runs(flow=[flow["id"]]))[:20000]:
        try:
            run = openml.runs.get_run(run_id)
            if run.task_id not in tasks:  # Check if we already downloaded task or dataset
                tasks[run.task_id] = openml.tasks.get_task(run.task_id)
            if tasks[run.task_id].dataset_id not in datasets:
                datasets[tasks[
                    run.task_id].dataset_id] = openml.datasets.get_dataset(
                        tasks[run.task_id].dataset_id)
            flow_runs.append({
                "accuracy": run.evaluations["predictive_accuracy"],
                "algorithm": flow["label"]
            } | datasets[tasks[run.task_id].dataset_id].qualities)
        except:
            continue
    return flow_runs


if __name__ == "__main__":
    labeled_flows = {}
    with jsonlines.open('flow_classification.jsonl', 'r') as f:
        for obj in f:
            labeled_flows[obj['text']] = obj
    results = p_map(download_runs, labeled_flows.values())
    flattened_results = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flattened_results)
    df.to_csv("flow_runs.csv")