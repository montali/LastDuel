import openml
from tqdm import tqdm
import pickle
from random import shuffle
from collections import defaultdict
from multiprocessing import Pool, RLock, freeze_support
from tqdm.contrib.concurrent import process_map
from p_tqdm import p_map

is_ensemble = {
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
runs_with_mlp = openml.runs.list_runs(flow=[1820])
with open("runs_with_mlp.pickle", "wb") as handle:
    pickle.dump(runs_with_mlp, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Downloaded runs with MLP\nNow downloading runs for these.")


def get_tasks(run):
    try:
        return openml.tasks.get_task(run["task_id"])
    except Exception as e:
        print(e)
        return None


tasks = p_map(get_tasks, runs_with_mlp.values())
with open("tasks.pickle", "wb") as handle:
    pickle.dump(tasks, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Downloaded runs. Now finding performances for each task.")


def find_performances_for_task(task):
    runs_for_task = []
    checked_flows = {}
    non_interesting_flows = set()
    runs = 0
    list_runs = openml.runs.list_runs(task=[task.task_id]).items()
    for run_id, run in list_runs:
        if runs > 2000:  # Don't get more than 2000 runs as it's too many
            break
        if (
            run["flow_id"] in non_interesting_flows
        ):  # Already checked this flow and it's not ensemble or mlp
            continue
        elif (
            run["flow_id"] in checked_flows
        ):  # Already checked this flow and it's ensemble/mlp, so it's interesting
            runs_for_task.append(
                (
                    openml.runs.get_run(run_id).evaluations["predictive_accuracy"],
                    checked_flows[run["flow_id"]],
                )
            )
        else:
            flow_name = openml.flows.get_flow(run["flow_id"]).name.lower()
            ensemble_found = [
                ensemble
                for method, ensemble in is_ensemble.items()
                if method in flow_name
            ]
            if ensemble_found:  # if the method used is an ensemble or NN
                runs_for_task.append(
                    (
                        openml.runs.get_run(run_id).evaluations["predictive_accuracy"],
                        ensemble_found[0],
                    )
                )
                checked_flows[run["flow_id"]] = ensemble_found[0]
            else:
                non_interesting_flows.add(run["flow_id"])
        runs += 1
    return runs_for_task


performances = p_map(find_performances_for_task, tasks)
