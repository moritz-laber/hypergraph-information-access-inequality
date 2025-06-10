"""
Simulation 

Read in a specified json file with parameter values to generate a hypergraph and run a set of simulations and output
Throughout this file, Group 0 is the majority group and Group 1 is the minority group. 


Parameters:

General:
    `num_runs`: int, number of runs on a single hypergraph with a single seeding condition, default 10
    `num_seed_cond`: int, number of seeding conditions to simulate on a single hypergraph, default 10
    `num_hg`: int, number of hypergraphs to simulate, default 10

Hypergraph:
    `num_nodes`: int, number of nodes in the hypergraph, default 10_000
    `p_m`: float, fraction of nodes that are minority group (group 1), 0.5
    `edge_size`: list of ints, edge sizes to simulate, default [2, 3, 4]
    `edge_counts`: dict of dict of type t (group 0) edge counts for each edge size, default None
    `hg_seed`: int, seed for hypergraph, default 42
    `d_s`: str, degree sampler to use, default 'powerlaw'
    `degree_seed`: int, seed for degree sampler, default 17
    `k_bar`: float, average degree of nodes, default 11
    `gamma`: float, power law exponent, default 2.5
    `h`: str, homophily pattern to use

Spreading:
    `nu`: float, nonlinearity parameters, default (1.0, 1.0)
    `lambda`: tuple, spreading rates, default (0.01,0.01)
    `num_seed`: int, number of seeds, default 4
    `seed_strategy`: string, strategy to select seed, default 'random'
    `sp_m`: float, fraction of seeds that are minority group (group 1), default 0.5
"""

from HyperGraph import * 
from HyperGraphHelper import *
from generate_params import *
import numpy as np
import json
import pickle as pkl
import os
import time
import pandas as pd
from typing import Dict, Tuple, List, Set
import tqdm

# GLOBAL PARAMETERS #
NUM_RUNS = 1                # number of runs on a single hypergraph with a single seeding condition
NUM_HG = 1000               # number of hypergraphs to create for each parameter configuration
NUM_NODES = 10_000          # number of nodes in the hypergraph
EDGE_SIZE = [2, 3, 4]       # edge sizes in the hypergraph
NUM_SEED = 4                # number of seed nodes
NUM_SEED_COND = 1           # number of repeated seedings on the same graph

HG_SEED = 42                # starting seed for hypergraph generation
DEGREE_SEED = 17            # seeds for the degree sampler
FAST = True                 # whether to skip sorting the nodes in each egdge when constructing the hypergraph, which is not necessary in conjunction with CSCM
SAMPLABLE_SET = True        # whetehr to use samplable set in the simulation


def get_params(file_path:str):
    """
    Read in parameters from json.
    Parameters:
        file_path: string, path to json file
    Returns:
        params: dict, dictionary of parameters
    """

    # change to the directory with the json file
    with open(f'{file_path}params.json', 'r') as f:
        params = json.load(f)
    return params


def get_hypergraph(file_path:str, i:int, path_to_hg:str=None):
    """
    Read in a hypergraph from a file.
    Parameters:
        file_path: string, path to the file
        i: int, index of the hypergraph
        path_to_hg: str, path to hypergraph
    Returns:
        edges: list of list of int, edges of the hypergraph
        group: dict, group of each node
        kappa: dict, degree of each node
    """

    if path_to_hg:
        with open(path_to_hg, 'rb') as f:
            hg = pkl.load(f)
            edges = hg.edges
            group = hg.group
            kappa = hg.degree
    else:
        with open(f'{file_path}hg{i}.pkl', 'rb') as f:
            edges, group, kappa = pkl.load(f)
    return edges, group, kappa


def run(params, file_path, hypergraphs_only=False, read_hypergraph=False, include_pairwise=False, seed_test=False, real_world=False, real_pred_gender=False):
    """
    Run simulations with the given parameters.
    Parameters:
        params: dict, dictionary of parameters
        file_path: string, path to save the results
        hypergraphs_only: bool, whether to only generate the hypergraphs
        read_hypergraph: bool, whether to read in the hypergraphs from the file
        include_pairwise: bool, whether to include projected graph experiments
        seed_test: bool, whether to run the seed test with hypercore and k-core seeding
        real_world: bool, whether to run the simulations on real-world data
        real_pred_gender: bool, whether the real-world data uses predicted gender
    Returns:None
    """
    # hypergraph params
    num_runs = params.get('num_runs', 10)
    num_seed_cond = params.get('num_seed_cond', 10)
    num_hg = params.get('num_hg', 10)
    num_nodes = params.get('num_nodes', 10_000)
    p_m = params.get('p_m', 0.5)
    edge_size = params.get('edge_size', [2, 3, 4])
    edge_counts = params.get('edge_counts', None)
    hg_seed = params.get('hg_seed', 42)
    d_s = params.get('d_s', 'powerlaw')
    gamma = params.get('gamma', 3.0)
    kbar = params.get('kbar', 11)
    degree_seed = params.get('degree_seed', 17)

    if real_world:
        original_path = params.get('hg_file_path', '')
    
    # spreading params
    nu = params.get('nu', (1.0, 1.0))
    lam = params.get('lam', (0.01, 0.01))
    num_seed = params.get('num_seed', 4)
    sp_m = params.get('sp_m', 0.5)
    seed_strategies = [params.get('seed_strategy', 'random')]
    
    if seed_test:
        seed_strategies = ['degree', 'hyper_core', 'k_core', 'random']
    # make all the edge_counts keys ints and the keys of the values ints
    if not real_world:
        edge_counts = {int(k): {int(kk): v for kk, v in v.items()} for k, v in edge_counts.items()}
    n1 = int(num_nodes * p_m)
    n0 = num_nodes - n1
    # set up dictionary for seed generation time
    time_data = {'fp': [], 'hg': [], 'degree': [], 'hyper_core': [], 'k_core': [], 'random': []}

    # run for a num_hg hypergraphs
    for i in tqdm.tqdm(range(num_hg)):
        if real_world:
            if real_pred_gender:
                path_to_hg = original_path + str(i) + '.pkl'
            else:
                path_to_hg = original_path

        # generate the degree sampler
        degree_seed += 1
        if real_world:
            pass
        elif d_s == 'powerlaw':
            degree_rng = np.random.default_rng(degree_seed)
            degree_sampler = lambda n, kbar, gamma : ((gamma - 2.)/(gamma - 1.))*kbar*(1 + degree_rng.pareto(gamma - 1, size=n))
            params = (gamma,)
        elif d_s == 'poisson':
            degree_rng = np.random.default_rng(degree_seed)
            degree_sampler = lambda n, kbar : degree_rng.poisson(kbar, size=n)
            params = ()
        else:
            raise ValueError('Invalid degree sampler type')
        
        if read_hypergraph: # read in hypergraph
            if real_world:
                edges, group, kappa = get_hypergraph(file_path, i, path_to_hg)
            else:
                edges, group, kappa = get_hypergraph(file_path, i)
        else: # generate hypergraph
            edges, group, kappa = CSCM([n0, n1], edge_counts, degree_sampler, params=params, seed=hg_seed*i)
        
        if hypergraphs_only: # save only the hypergraph
            with open(f'{file_path}hg{i}.pkl', 'wb') as f:
                pkl.dump((edges, group, kappa), f)
            continue

        hASI = HyperGraphAsymmetricSI(nodes=np.arange(0, num_nodes), edges=edges, group=group, nus=np.array([[nu[0],nu[1]],[nu[1],nu[0]]]), lams=np.array([[lam[0],lam[1]],[lam[1],lam[0]]]), fast=FAST, samplable_set=SAMPLABLE_SET)
        # project hypergraph to pairwise graph
        if include_pairwise:
            H = HyperGraph(nodes=np.arange(0, num_nodes), edges=edges, group=group, fast=FAST)
            G = H.clique_projection()

            hASI_pairwise = HyperGraphAsymmetricSI(nodes=np.arange(0, G.N), edges=G.edges, group=G.group, nus=np.array([[nu[0],nu[1]],[nu[1],nu[0]]]), lams=np.array([[lam[0],lam[1]],[lam[1],lam[0]]]), fast=FAST)

        # run for each type of seed
        time_data['fp'].append(file_path)
        time_data['hg'].append(i)
        for seed_strategy in seed_strategies:
            # run for num_seed_cond seeding conditions
            for j in range(num_seed_cond):
                # get the seeds for the simulation
                start = time.time()
                seeds = select_seeds(seed_strategy, hASI, num_seed, sp_m)
                end = time.time()
                elapsed = end-start
                
                time_data[seed_strategy].append(elapsed)
                run_simulation(seeds, hASI, i, seed_strategy, j, num_runs=num_runs, file_path=file_path, seed_test=seed_test)
                if include_pairwise:
                    run_simulation(seeds, hASI_pairwise, i, seed_strategy, j, num_runs=num_runs, file_path=file_path,
                                   pairwise=True)

    # save seed computation times
    if seed_test:
        seed_times = pd.DataFrame(data=time_data)
        fp = file_path + 'seed_times.csv'
        seed_times.to_csv(fp, index=False)


def run_simulation(seeds:ArrayLike, hASI:HyperGraph, iter:int, seed_strategy:str, j:int, num_runs:int=10, file_path='data/', pairwise=False, seed_test=False):
    """
    Run the simulation and parse the results.
    Parameters:
        seeds: list of int, seeds for the simulation
        hASI: HyperGraphAsymmetricSI, hypergraph for the simulation
        num_runs: int, number of runs for the simulation
        file_path: string, path to save the results
        pairwise: bool, whether to run projected graph experiments
        seed_test: bool, whether to save the seed test results
    Returns:
        None.
        Saves: dict of arrays, where the keys are the results and 
        the values are arrays with 3 columns: node label, time of infection, group of node 
    """

    results = {}
    for i in tqdm.tqdm(range(num_runs)):
        # set the conditions
        hASI.set_initial_condition(seeds)
        # run the simulation
        _, time_seq, node_seq, group_seq, event_type_seq, edge_size_seq, edge_type_seq = hASI.simulate(maxiter=5000000, verbose=False)
        # add the seeds to the beginning of node_seq, time_seq, group_seq
        node_seq = np.concatenate([np.array(seeds), node_seq[1:]])
        time_seq = np.concatenate([np.zeros(len(seeds)), time_seq[1:]])
        group_seq = np.concatenate([np.array([hASI.group[s] for s in seeds]), group_seq[1:]])
        event_type_seq = np.concatenate([np.array([-1 for s in seeds]), event_type_seq[1:]])
        edge_size_seq = np.concatenate([np.array([-1 for s in seeds]), edge_size_seq[1:]])
        edge_type_seq = np.concatenate([np.array([-1 for s in seeds]), edge_type_seq[1:]])
        results[i] = np.array([node_seq, time_seq, group_seq, event_type_seq, edge_size_seq, edge_type_seq]).T

    # save the results
    if pairwise:
        fp = f'{file_path}GRAPH_TEST_results_hg{iter}_{seed_strategy}{j}.pkl'
    elif seed_test and (seed_strategy == 'h_core' or seed_strategy == 'k_core'):
        fp = f'{file_path}SEED_TEST_results_hg{iter}_{seed_strategy}{j}.pkl'
    else:
        fp = f'{file_path}results_hg{iter}_{seed_strategy}{j}.pkl'

    with open(fp, 'wb') as f:
        pkl.dump(results, f)


def run_all(file_path='./', hypergraphs_only=False, read_hypergraph=False, include_pairwise=False, seed_test=False, real_world=False, real_pred_gender=False):
    """
    Run all the simulations and writes the results to the file with the params.json
    Parameters:
        file_path: string, path to the params files
        hypergraphs_only: bool, whether to only generate the hypergraphs 
        read_hypergraph: bool, whether to read in the hypergraphs from the file
        include_pairwise: bool, whether to run the projected graph experiments
        seed_test: bool, whether to run the seed test
        real_world: bool, whether to run the simulations on real-world data
        real_pred_gender: bool, whether the real-world data uses predicted gender
    """

    # get the directories in the file_path
    dirs = os.listdir(file_path)

    for dir in tqdm.tqdm(dirs):
        print(f'Running simulations for {dir}')
        file_path_dir = f'{file_path}{dir}/'
        params = get_params(file_path_dir)
        run(params, file_path_dir, hypergraphs_only=hypergraphs_only, read_hypergraph=read_hypergraph, include_pairwise=include_pairwise, seed_test=seed_test, real_world=real_world, real_pred_gender=real_pred_gender)


def main():

    # Synthetic simulations

    # step 1: generate_params
    generate_params()

    # step 2: generate hypergraphs and run simulation
    #   Note: setting hypergraphs_only to True will generate hypergraphs without running contagion
    run_all(file_path= 'data/', hypergraphs_only=False, read_hypergraph=False, include_pairwise=False, seed_test=False)


    # Real-world simulations

    # non-predicted gender hypergraphs
    datasets = ['hospital', 'highschool', 'primaryschool', 'housebills', 'senatebills']

    for dataset in datasets:
        # step 1: generate_params
        generate_params(file_path=f'data/real_world/{dataset}/simulations/', real_world=dataset, hg_file_path=f'data/real_world/{dataset}/hypergraphs/{dataset}_lcc_hg.pkl', real_pred_gender=False)

        # step 2: run simulation
        #   Note: must set read_hypergraph to True for real_wold data
        #   Note: real_world must be set to True
        run_all(file_path=f'data/real_world/{dataset}/simulations/', hypergraphs_only=False, read_hypergraph=True, include_pairwise=False, seed_test=False, real_world=True, real_pred_gender=False)

    # predicted gender hypergraphs
    datasets = ['housebillsgender', 'senatebills', 'dblp', 'aps']

    for dataset in datasets:
        # step 1: generate_params
        generate_params(file_path=f'data/real_world/{dataset}/simulations_genderapi/', real_world=dataset, hg_file_path=f'data/real_world/{dataset}/hypergraphs_genderapi/hg_lcc_', real_pred_gender=True)
        generate_params(file_path=f'data/real_world/{dataset}/simulations_genderizerio/', real_world=dataset, hg_file_path=f'data/real_world/{dataset}/hypergraphs_genderizerio/hg_lcc_', real_pred_gender=True)

        # step 2: run simulation
        #   Note: must set read_hypergraph to True for real_wold data
        #   Note: real_world must be set to True
        #   Note: real_pred_gender must be set to True
        run_all(file_path=f'data/real_world/{dataset}/simulations_genderapi/', hypergraphs_only=False, read_hypergraph=True, include_pairwise=False, seed_test=False, real_world=True, real_pred_gender=True)
        run_all(file_path=f'data/real_world/{dataset}/simulations_genderizerio/', hypergraphs_only=False, read_hypergraph=True, include_pairwise=False, seed_test=False, real_world=True, real_pred_gender=True)


if __name__=='__main__':
    main()

