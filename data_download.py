import openml
from tqdm import tqdm
import pickle
from random import shuffle
from collections import defaultdict
from multiprocessing import Pool, RLock, freeze_support
from tqdm.contrib.concurrent import process_map
from p_tqdm import p_map
import pandas as pd
import numpy as np
from os import path
from openml.runs import OpenMLRun
from openml.tasks import OpenMLTask
from openml.flows import OpenMLFlow
from typing import List, Tuple, Optional


def get_tasks(run: OpenMLRun) -> Optional[OpenMLTask]:
    """
    Downloads the OpenMLTask objects for a given run.
    Returns none when the task is not found: this requires a successive cleaning of the results.

    Args:
        run: Run to get the tasks for

    Returns:
        Task for the run
    """
    try:
        return openml.tasks.get_task(run["task_id"])
    except Exception as e:
        return None


def find_performances_for_task(task: OpenMLTask) -> List[Tuple[float, bool]]:
    """
    Gets the performance values for a given task, consisting in tuples (accuracy, is_ensemble),
    with is_ensemble being True for ensembles, False for Neural Networks

    Args:
        task: Task to get the performances for

    Returns:
        Performances for the task
    """

    runs_for_task = []
    checked_flows = {}
    non_interesting_flows = set()
    runs = 0
    list_runs = openml.runs.list_runs(task=[task.task_id]).items()
    for run_id, run in list_runs:
        if runs > 2000:  # Don"t get more than 2000 runs as it"s too many
            break
        # Already checked this flow and it"s not ensemble or mlp
        if (run["flow_id"] in non_interesting_flows):
            continue
        elif (run["flow_id"] in checked_flows):
            # Already checked this flow and it"s ensemble/mlp, so it"s interesting
            try:
                runs_for_task.append((
                    openml.runs.get_run(
                        run_id).evaluations["predictive_accuracy"],
                    checked_flows[run["flow_id"]],
                ))
            except:
                continue
        else:
            flow_name = openml.flows.get_flow(run["flow_id"]).name.lower()
            ensemble_found = [
                ensemble for method, ensemble in is_ensemble.items()
                if method in flow_name
            ]
            if ensemble_found:  # if the method used is an ensemble or NN
                try:
                    runs_for_task.append((
                        openml.runs.get_run(
                            run_id).evaluations["predictive_accuracy"],
                        ensemble_found[0],
                    ))
                except:
                    continue
                checked_flows[run["flow_id"]] = ensemble_found[0]
            else:
                non_interesting_flows.add(run["flow_id"])
        runs += 1
    return runs_for_task


if __name__ == "__main__":
    is_ensemble = {  # Dictionary containing keywords for ensemble/NN
        "boost": True,
        "ada": True,
        "forest": True,
        "ensemble": True,
        "bag": True,
        "nn": False,
        "nnet": False,
        "mlp": False,
        "multilayerperceptron": False,
    }
    if path.exists("data/flows_with_mlp.pkl"):
        with open("data/flows_with_mlp.pkl", "rb") as f:
            flows_with_mlp = pickle.load(f)
        print("Loaded data/flows_with_mlp.pkl")
    else:
        nnet_names = {name for name, value in is_ensemble.items() if not value}
        # Get flows with nn or ensemble
        flows_with_mlp = []
        for flow in openml.flows.list_flows():
            flow_name = openml.flows.get_flow(flow).name
            for name in nnet_names:
                if name in flow_name.lower():
                    flows_with_mlp.append(flow)
        print(f"Found {len(flows_with_mlp)} flows with mlp")

    if path.exists("data/runs_with_mlp.pkl"):
        with open("data/runs_with_mlp.pkl", "rb") as f:
            runs_with_mlp = pickle.load(f)
        print("Loaded data/runs_with_mlp.pkl")
    else:
        runs_with_mlp = []
        for flow in tqdm(flows_with_mlp):
            runs_with_mlp += openml.runs.list_runs(flow=[flow]).values()
        with open("data/runs_with_mlp.pkl", "wb") as handle:
            pickle.dump(runs_with_mlp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            "Downloaded runs with MLP\nNow downloading data and runs for these."
        )

    if path.exists("data/tasks.pkl"):
        with open("data/tasks.pkl", "rb") as f:
            tasks = pickle.load(f)
        print("Loaded data/tasks.pkl")
    else:
        tasks = p_map(get_tasks, runs_with_mlp)
        tasks = [task for task in tasks if task]  # Delete missing
        with open("data/tasks.pkl", "wb") as handle:
            pickle.dump(tasks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if path.exists("data/data_for_task.pkl"):
        with open("data/data_for_task.pkl", "rb") as f:
            data_for_task = pickle.load(f)
        print("Loaded data/data_for_task.pkl")
    else:
        data_for_task = {}
        for task in tqdm(tasks):
            data_for_task[task.task_id] = openml.datasets.get_dataset(
                task.dataset_id)
        with open("data/data_for_task.pkl", "wb") as handle:
            pickle.dump(data_for_task, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Downloaded runs. Now finding performances for each task.")

    performances = []
    if path.exists("data/performances.pkl"):
        with open("data/performances.pkl", "rb") as f:
            performances = pickle.load(f)
        print("Loaded data/performances.pkl")
    if len(performances) < len(tasks):
        for i in range(len(performances), len(tasks), 100):
            performances += p_map(find_performances_for_task, tasks[i:i + 100])
            with open("data/performances.pkl", "wb") as handle:
                pickle.dump(performances,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    # Generate a Pandas dataframe with the data we collected

    rows = []
    for i, task in enumerate(tasks[:3300]):
        average_ensemble = np.mean(
            [run for run, is_ensemble in performances[i] if is_ensemble])
        average_mlp = np.mean(
            [run for run, is_ensemble in performances[i] if not is_ensemble])
        this_task = {
            "id": task.task_id,
            "average_ensemble": average_ensemble,
            "average_mlp": average_mlp,
            "ensemble_mlp_diffn": average_ensemble - average_mlp,
        }
        rows.append({**this_task, **data_for_task[task.task_id].qualities})

    dataset = pd.DataFrame(rows).dropna(
        subset=["ensemble_mlp_diffn"]).interpolate()
    dataset.to_csv("data_for_task.csv")
