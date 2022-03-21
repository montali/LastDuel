from os import path
import numpy as np
import pickle
import pandas as pd

if __name__ == "__main__":
    if path.exists("data/tasks.pkl"):
        with open("data/tasks.pkl", "rb") as f:
            tasks = pickle.load(f)
        print("Loaded data/tasks.pkl")

    if path.exists("data/data_for_task.pkl"):
        with open("data/data_for_task.pkl", "rb") as f:
            data_for_task = pickle.load(f)
        print("Loaded data/data_for_task.pkl")

    if path.exists("data/performances.pkl"):
        with open("data/performances.pkl", "rb") as f:
            performances = pickle.load(f)
        print(
            f"Loaded data/performances.pkl: there are {len(performances)} performances"
        )
    rows = []
    for i, task in enumerate(tasks[:len(performances)]):
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
