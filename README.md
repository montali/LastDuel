# LastDuel

Artificial Intelligence MSc thesis: trying to predict when ensembles perform better/worse than Neural Networks through OpenML data.

## Introduction

The automation of machine learning tasks represents one of the most interesting, yet complex tasks that could push the implementation of ML in more and more fields. A lot of the time spent by machine learning engineers is spent picking and optimising models, a task that's not too difficult to be automated. What's more interesting, though, is trying to understand the patterns between the performance of a model, the task, and the model itself. This is among the main concerns of **meta-learning**, or in other words, _learning how to learn_.

## Ensembles and neural networks

While neural networks have been the bleeding-edge standard for machine learning in the last years, pushed by the incredible improvement in terms of parallelized computational power that we witnessed in the last decade, we have recently seen an enormous increase in the usage of **ensembling techniques**. These methods often make use of simpler learners, like decision trees, and merge the results of multiple of these models. By implying that these models are wrong, but are wrong in different ways, if we consider all the results, the accuracy improves. Tools like XGBoost, CatBoost or RandomForests have become the industry standard for tabular tasks, being way faster, and sometimes more accurate, than neural networks. We expect neural networks to be able to outperform ensembles in almost every case, but this would imply that we're always able to pick optimal parameters for the network, which is rarely true. In a real-world comparison, we can take data from **OpenML** to try and understand how ensembles perform in daily tasks that are submitted to the database.

## OpenML

**OpenML** is an open and collaborative automated machine learning environment. It provides an enormous quantity of data about machine learning tasks, datasets and runs, allowing us to perform analysis on how these algorithms learn. It operates on a number of core concepts which are crucial to understand:

- **Datasets**: datasets are pretty straight-forward. They simply consist of a number of rows, also called instances, usually in tabular form;
- **Tasks**: a task consists of a dataset, together with a machine learning task to perform, such as classification or clustering and an evaluation method. For supervised tasks, this also specifies the target column in the data;
- **Flows**: a flow identifies a particular machine learning algorithm from a particular library or framework such as Weka, mlr or scikit-learn. It should at least contain a name, details about the workbench and its version and a list of settable hyperparameters. Ideally, the appropriate workbench can deserialize it again (the algorithm, not the model);
- **Runs**: a run is a particular flow, that is algorithm, with a particular parameter setting, applied to a particular task.

Everyone can contribute to the OpenML data, and more than 10 millions runs are available at the moment. The nicest feature of OpenML is that it automatically computes **meta-features** for the datasets, such as skew, performance on kNN and many more. These data will then be used to perform the actual experiment.

## Data retrieval

OpenML comes with an API (wrapped in multiple languages) that allows us to download the data easily and straightforwardly. To retrieve the data for our experiment, the following steps are required:

- Getting the runs in which a NN was used
- Getting the tasks for these runs
- Getting the meta-features for the datasets linked to these tasks
- Averaging the results for ensemble and MLP methods and saving these, together with the meta-features, in a DataFrame

All these steps are done in the [data_download.py](https://github.com/montali/LastDuel/blob/main/data_download.py) script.

## Ensembles vs Neural Networks model

A first, interesting model to train would be one that's able to infer the difference in performance between ensembles and neural networks, given a dataset. To do so, we can generate a target value being the difference between the two averages, and train a regressor (both an ensemble and a neural network🥴). This is done in the [regression notebook](https://github.com/montali/LastDuel/blob/main/regression.ipynb).

## Generalized model

After working on a comparison between ensembles and NNs, it became clear how it is indeed possible to get insights about algorithm performances basing on the task metafeatures. To do this, a dataset (`flow_runs.csv`) has been generated, labeling the algorithms (e.g. Neural Network, Ensemble, Naive Bayes, Logistic Regression...) with OpenAI's APIs (script in `flow_labeling.py`, results in `flow_classification.jsonl`), then training a neural network to predict the accuracy of the models. The R2 score that was obtained on the test set (which has been split using the tasks, so that the predicted tasks are never seen before) was around 0.8.
