# A module for data managements and post-processing for the experiments
# ==========================================================


import numpy as np
import torch as t
import pandas as pd

import os
from glob import glob
import random
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTS_BATCH_SIZE = 25





# Managing datasets and activations
# ---------------------------------

def collect_acts(dataset_name, model_family, model_size,
                  model_type, layer, center=True, scale=False, device='cpu'):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    directory = os.path.join(ROOT, 'acts', model_family, model_size, model_type, dataset_name)
    activation_files = glob(os.path.join(directory, f'layer_{layer}_*.pt'))
    acts = [t.load(os.path.join(directory, f'layer_{layer}_{i}.pt'), map_location=device) for i in range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)] 
    try:
        acts = t.cat(acts, dim=0).to(device)
    except:
        raise Exception("No activation vectors could be found for the dataset " 
                        + dataset_name + ". Please generate them first using generate_acts.")
    if center:
        acts = acts - t.mean(acts, dim=0)
    if scale:
        acts = acts / t.std(acts, dim=0)
    return acts



def cat_data(d):
    """
    Given a dict of datasets (possible recursively nested), returns the concatenated activations and labels.
    """
    all_acts, all_labels = [], []
    for dataset in d:
        if isinstance(d[dataset], dict):
            if len(d[dataset]) != 0: # disregard empty dicts
                acts, labels = cat_data(d[dataset])
                all_acts.append(acts), all_labels.append(labels)
        else:
            acts, labels = d[dataset]
            all_acts.append(acts), all_labels.append(labels)
    try:
        acts, labels = t.cat(all_acts, dim=0), t.cat(all_labels, dim=0)
    except:
        raise Exception("No activation vectors could be found for this dataset. Please generate them first using generate_acts.")
    return acts, labels



class DataManager:
    """
    Class for storing activations and labels from datasets of statements.
    """
    
    def __init__(self):
        self.data = {
            'train' : {},
            'val' : {}
        } # dictionary of datasets
        self.proj = None # projection matrix for dimensionality reduction
    

    def add_dataset(self, dataset_name, model_family, model_size, model_type, layer,
                     label='label', split=None, seed=None, center=True, scale=False, device='cpu'):
        """
        Add a dataset to the DataManager.
        label : which column of the csv file to use as the labels.
        If split is not None, gives the train/val split proportion. Uses seed for reproducibility.
        """
        acts = collect_acts(dataset_name, model_family, model_size, model_type,
                             layer, center=center, scale=scale, device=device)
        df = pd.read_csv(os.path.join(ROOT, 'datasets', f'{dataset_name}.csv'))
        labels = t.Tensor(df[label].values).to(device)

        if split is None:
            self.data[dataset_name] = acts, labels

        if split is not None:
            assert 0 <= split and split <= 1
            if seed is None:
                seed = random.randint(0, 1000)
            t.manual_seed(seed)
            train = t.randperm(len(df)) < int(split * len(df))
            val = ~train
            self.data['train'][dataset_name] = acts[train], labels[train]
            self.data['val'][dataset_name] = acts[val], labels[val]


    def get(self, datasets):
        """
        Output the concatenated activations and labels for the specified datasets.
        datasets : can be 'all', 'train', 'val', a list of dataset names, or a single dataset name.
        If proj, projects the activations using the projection matrix.
        """
        if datasets == 'all':
            data_dict = self.data
        elif datasets == 'train':
            data_dict = self.data['train']
        elif datasets == 'val':
            data_dict = self.data['val']
        elif isinstance(datasets, list):
            data_dict = {}
            for dataset in datasets:
                if dataset[-6:] == ".train":
                    data_dict[dataset] = self.data['train'][dataset[:-6]]
                elif dataset[-4:] == ".val":
                    data_dict[dataset] = self.data['val'][dataset[:-4]]
                else:
                    data_dict[dataset] = self.data[dataset]
        elif isinstance(datasets, str):
            data_dict = {datasets : self.data[datasets]}
        else:
            raise ValueError(f"datasets must be 'all', 'train', 'val', a list of dataset names, or a single dataset name, not {datasets}")
        acts, labels = cat_data(data_dict)
        # if proj and self.proj is not None:
        #     acts = t.mm(acts, self.proj)
        return acts, labels



def dataset_sizes(datasets):
    """
    Computes the size of each dataset, i.e. the number of statements.
    Input: array of strings that are the names of the datasets
    Output: dictionary, keys are the dataset names and values the number of statements
    """
    dataset_sizes_dict = {}
    for dataset in datasets:
        file_path = 'datasets/' + dataset + '.csv'
        with open(file_path, 'r') as file:
            line_count = sum(1 for line in file)
        dataset_sizes_dict[dataset] = line_count - 1
    return dataset_sizes_dict



def collect_training_data(dataset_names, train_set_sizes, model_family, model_size
                          , model_type, layer, **kwargs):
    """
    Takes as input the names of datasets in the format
    [affirmative_dataset1, negated_dataset1, affirmative_dataset2, negated_dataset2, ...]
    and returns a balanced training dataset of centered activations, activations, labels and polarities
    """
    all_acts_centered, all_acts, all_labels, all_polarities = [], [], [], []
    
    for dataset_name in dataset_names:
        dm = DataManager()
        dm.add_dataset(dataset_name, model_family, model_size, model_type, layer, split=None, center=False, device='cpu')
        acts, labels = dm.data[dataset_name]
        
        polarity = -1.0 if 'neg_' in dataset_name else 1.0
        polarities = t.full((labels.shape[0],), polarity)

        # balance the training dataset by including an equal number of activations from each dataset
        # choose the same subset of statements for affirmative and negated version of the dataset
        if 'neg_' not in dataset_name:
            rand_subset = np.random.choice(acts.shape[0], min(train_set_sizes.values()), replace=False)
        
        all_acts_centered.append(acts[rand_subset, :] - t.mean(acts[rand_subset, :], dim=0))
        all_acts.append(acts[rand_subset, :])
        all_labels.append(labels[rand_subset])
        all_polarities.append(polarities[rand_subset])

    return map(t.cat, (all_acts_centered, all_acts, all_labels, all_polarities))





# Post-processing of the experimental results
# -------------------------------------------

def compute_statistics(results:dict) -> dict:
    """
    Computes the mean and standard deviation of the classification accuracies for each probe and each dataset.

    Args:
        - results[probe_type : class.probes][validation_dataset : str] : list[float] = classification accuracies for each training iteration

    Output:
        - stats[probe_type : class.probes] : dict = {"means":dict, "stds":dict} 
            - means[dataset_name : str] : float = mean of accuracy
            - stds[dataset_name : str] : float = std of accuracy
    """
    
    stats = {}
    for key in results:
        means = {dataset: np.mean(values) for dataset, values in results[key].items()}
        stds = {dataset: np.std(values) for dataset, values in results[key].items()}
        stats[key] = {'mean': means, 'std': stds}
    
    return stats



def compute_average_accuracies(results:dict, num_iter:int) -> dict:
    """
    Computes the mean accuracy of each probe across all datasets and also the standard deviation of this mean aross all training iterations.
    This is done by taking the mean of the accuracies across all datasets for each training iteration, and then taking the standard deviation of these means.
    The final mean accuracy is computed by taking the mean of the means across all datasets.
    
    Args:
        - results[probe_type : class.probes][validation_dataset : str] : list[float] = classification accuracies for each training iteration
        - num_iter : int = number of training iterations
    
    Output:
        - probe_stats[probe_type : class.probes] : dict = {"mean":float, "std_dev":float}
            - mean = mean of accuracy of probe_type across all datasets and iterations
            - std_dev = standard deviation of the mean accuracy for each dataset
    """

    probe_stats = {}

    for probe_type in results:
        overall_means = []
        
        for i in range(num_iter):
            # Calculate mean accuracy for each dataset in this iteration
            iteration_means = [results[probe_type][dataset][i] for dataset in results[probe_type]]
            overall_means.append(np.mean(iteration_means))
        
        overall_means = np.array(overall_means)
        final_mean = np.mean(overall_means)
        std_dev = np.std(overall_means)
        
        probe_stats[probe_type] = {'mean':final_mean, 'std_dev':std_dev}
    
    return probe_stats



def show_and_save_results (probe_accuracies:dict, dataset_accuracies:dict, meta_info:dict, save=True) -> None:
    """
    Print and save the results of the probe accuracy statistics.
    The saved file will have the following format: 
    `./results/<notebook_name>--<experiment_name>--<full_model_name>.json`
    
    Inputs:
    - probe_accuracies : dict = overall accuracy statistics for each probe type
        probe_accuracies[probe_type : class.probes] : dict = {"mean":float, "std_dev":float}
            - mean = mean of accuracy of probe_type across all datasets and iterations
            - std_dev = standard deviation of the mean accuracy for each dataset
    - dataset_accuracies : dict = accuracy statistics for each probe type and validation dataset
        dataset_accuracies[probe_type : class.probes] : dict = {"means":dict, "stds":dict, "num_iter":int} 
            - means[dataset_name : str] : float = mean of accuracy
            - stds[dataset_name : str] : float = std of accuracy
            - num_iter : int = number of training iterations 
    - meta_info : dict = meta information about the experiment
        = {
            'notebook_name' : str,
            'model_name' : str,
            'layer' : int,
            'experiment_name' : str,
            'probe_names' : list[str],
            'validation_type' : str,
            'num_iterations' : int,
            'train_sets' : list[str],
            'val_sets' : list[str],
        }
    """

    # Print the results
    print(f"Results for the experiment '{meta_info['experiment_name']}' and the model '{meta_info['model_name']}' (layer {meta_info['layer']}):")
    print(f"    mean accuracy ± std deviation")
    print(f"------------------------------------")
    for probe_type, stats in probe_accuracies.items():
        print(f"{probe_type:<20}: {stats['mean']*100:4.1f} ± {stats['std_dev']*100:4.1f}%")
    print(f"------------------------------------")
    print("")


    # Save the results to a .json file in the results/ directory.
    if save:
        results = {
            'probe_accuracies':probe_accuracies, 'dataset_accuracies':dataset_accuracies, 'meta_info':meta_info,
        }
        os.makedirs("./results/", exist_ok=True)
        save_dir = f"./results/{meta_info['notebook_name']}--{meta_info['experiment_name']}--{meta_info['model_name']}.json" 
        with open(save_dir, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {save_dir}")



def load_results(notebook_name:str, experiment_name:str, full_model_name:str) -> tuple[dict]:
    """
    Load the experiment statistics (probe and dataset accuracies) from a .json file.
    The file is expected to be in the format: 
    `./results/<notebook_name>--<experiment_name>--<full_model_name>.json`.
    
    Arguments:
    - notebook_name = name of the notebook (e.g., "lie_detection").
    - experiment_name = name of the experiment (e.g., "unseen_topics").
    - full_model_name = name of the model (e.g., "Llama3_8B_chat").
    
    Returns:
    - results : tuple[dict]
        - probe_accuracies : dict = overall accuracy statistics for each probe type
            probe_accuracies[probe_type : class.probes] : dict = {"mean":float, "std_dev":float}
                - mean = mean of accuracy of probe_type across all datasets and iterations
                - std_dev = standard deviation of the mean accuracy for each dataset
        - dataset_accuracies : dict = accuracy statistics for each probe type and validation dataset
            dataset_accuracies[probe_type : class.probes] : dict = {"means":dict, "stds":dict, "num_iter":int} 
                - means[dataset_name : str] : float = mean of accuracy
                - stds[dataset_name : str] : float = std of accuracy
                - num_iter : int = number of training iterations 
        - meta_info: meta information about the experiment.
            dict = {
                'notebook_name' : str,
                'model_name' : str,
                'layer' : int,
                'experiment_name' : str,
                'probe_names' : list[str],
                'validation_type' : str,
                'num_iterations' : int,
                'train_sets' : list[str],
                'val_sets' : list[str],
                }
    """

    with open(f"./results/{notebook_name}--{experiment_name}--{full_model_name}.json", "r") as f:
        results = json.load(f)
    return results["probe_accuracies"], results["dataset_accuracies"], results["meta_info"]
