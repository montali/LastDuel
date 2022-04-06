"""Helps the user categorize flows into different categories.

This script will download the most popular flows from OpenML, then ask the user to categorize them.
Remember to upload the flow_classification_openai.jsonl file first.
"""

from cProfile import label
import openml
import openai
import jsonlines
import pickle
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut



openai.api_key_path = 'openai.key'
def save_flows(labeled_flows, flows_to_label):
    """
    It takes a dictionary of labeled flows and a set of flows to label, and saves them to disk
    
    Args:
      labeled_flows: a dictionary of labeled flows.
      flows_to_label: a list of flows that need to be labeled
    """
    with jsonlines.open('flow_classification.jsonl', mode='w') as f:
        for flow, flow_type in labeled_flows.items():
            f.write(flow_type | {'text': flow})
    with open('data/flows_to_label.pkl', 'wb') as f:
        pickle.dump(flows_to_label, f)

def load_flows():
    """
    It loads the flows that have been labeled and the flows that need to be labeled
    
    Returns:
      The dictionary of labeled flows, and the set of flows to label.
    """
    labeled_flows = {}
    with jsonlines.open('flow_classification.jsonl', 'r') as f:
        for obj in f:
            labeled_flows[obj['text']] = obj
    with open('data/flows_to_label.pkl', 'rb') as f:
        flows_to_label = pickle.load(f)
    return labeled_flows, flows_to_label

labeled_flows, flows_to_label = load_flows()
prompt = "Machine learning algorithm classifier. Given a class name, returns a type of ML algorithm among the following: NeuralNetwork, Ensemble, kNearestNeighbor, NaiveBayes, SupportVectorMachine.\n"
for name, data in labeled_flows.items():
    if len(prompt)>1000:
        break
    prompt += f"\nAlgorithm:{name}\nType:{data['label']}"

with tqdm(range(1000)) as t: # We process 1k flows per run
    for _ in t:
        to_label_id, to_label_name = flows_to_label.pop()
        try:
            l = func_timeout(5, lambda label_id: len(openml.runs.list_runs(flow=[label_id])), args=(to_label_id,))
        except FunctionTimedOut:
            print(f"{to_label_name} timed out")
            continue
        t.set_postfix(name=to_label_name[:15], runs=l)
        # Only do it for algos that were actually used
        if l > 1500:
            new_prompt = prompt + f"\nAlgorithm:{to_label_name}\nType:"
            labeled_flows[to_label_name] = {"label": openai.Completion.create(
                engine="text-davinci-001",
                    prompt=new_prompt,
                    temperature=0,
                    max_tokens=100,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )["choices"][0]["text"],
                "id": to_label_id}
            save_flows(labeled_flows, flows_to_label)



