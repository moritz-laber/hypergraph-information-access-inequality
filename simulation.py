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

def write_params(file_path:str, p_m:float, edge_counts:Dict[int, Dict[int,int]], nu:Tuple[float, float],
                 lam:Tuple[float,float], seed_strategy:str, homophily:str, d_s:str, kbar:float, gamma:float, sp_m:float,
                 real_world_hg:HyperGraph=None, hg_file_path:str='', real_pred_gender:bool=False):
    """
    Write parameters to a json file.
    Parameters:
        file_path: string, path to json file
        p_m: float, minority proportion
        edge_counts: Dict[int, Dict[int,int]], dictionary of edge counts
        nu: float, spreading nonlinearity parameters
        lam: float, spreading rate parameters
        seed_strategy: string, seeding strategy
        homophily: float, homophily value
        d_s: str, degree sampler to use
        kbar: float, average degree of nodes
        gamma: float, power law exponent
        sp_m: float, fraction of seeds that are minority group
        real_world: string, name of real-world hypergraph
        real_world_hg: HyperGraph, real-world hypergraph object
        hg_file_path: string, path to hypergraph object
        real_pred_gender: bool, whether the real world data uses predicted gender
    """
    # os.chdir(file_path)

    # overwrite constants if we have a real world hypergraph
    if real_world_hg:
        if real_pred_gender:
            num_hg = 100
            num_runs = 10
        else:
            num_runs = 1000
            num_hg = 1
        num_nodes = real_world_hg.N
        edge_size = list(real_world_hg.m.keys())
        num_seed = 1

        # constants first
        params = {
            'num_runs': num_runs,
            'num_seed_cond': NUM_SEED_COND,
            'num_hg': num_hg,
            'num_nodes': num_nodes,
            'p_m': p_m,
            'edge_size': edge_size,
            'hg_seed': HG_SEED,
            'd_s': d_s,
            'kbar': kbar,
            'gamma': gamma,
            'degree_seed': DEGREE_SEED,
            'nu': nu,
            'lam': lam,
            'num_seed': num_seed,
            'seed_strategy': seed_strategy,
            'sp_m': sp_m ,
            'hg_file_path': hg_file_path
        }
    else:
        # constants first
        params = {
            'num_runs': NUM_RUNS,
            'num_seed_cond': NUM_SEED_COND,
            'num_hg': NUM_HG,
            'num_nodes': NUM_NODES,
            'p_m': p_m,
            'edge_size': EDGE_SIZE,
            'edge_counts': edge_counts,
            'hg_seed': HG_SEED,
            'd_s': d_s,
            'kbar': kbar,
            'gamma': gamma,
            'degree_seed': DEGREE_SEED,
            'nu': nu,
            'lam': lam,
            'num_seed': NUM_SEED,
            'seed_strategy': seed_strategy,
            'sp_m': sp_m,
            'hg_file_path': hg_file_path
        }

    if real_world_hg:
        p_m = 'real'

    file_path = f'{file_path}pm{p_m}_spm{sp_m}_h{homophily}_ds{d_s}_nu{nu}_lam{lam}_ss{seed_strategy}/'

    # make the directory if it doesn't exist
    os.makedirs(file_path, exist_ok=True)
    # check to make sure the params.json file doesn't already exist
    if os.path.exists(f'{file_path}params.json'):
        print(f'File {file_path}params.json already exists')
    else:
        with open(f'{file_path}params.json', 'w') as f:
            json.dump(params, f)


def generate_params(file_path:str='data/', real_world:str=None, hg_file_path:str=None, real_pred_gender:bool=False,
                    nus:list=None, lams:list=None):
    """
    Generate the params files for all the different parameter combinations.
    Parameters:
        file_path: string, path to directory
        real_world: string, name of real-world hypergraph
        hg_file_path: string, path to real-world hypergraph
        real_pred_gender: boolean, whether real world data uses predicted gender
        nus: list of tuples, values of nu to use in simulations
        lams: list of tuples, values of lambda to use in limulations
    """
    p_ms = [0.25, 0.5]
    sp_ms = [0.0, 'prop', 1.0]  # use prop to set sp_m = p_m

    homophilies = ['weak', 'neutral', 'het', 'neutralweak3', 'weakneutral3', 'hetweak3', 'weakhet3']
    if not nus:
        nus = [(1.0, 1.0), (2.0, 2.0), (0.5, 0.5), (2.0, 0.5)]
    if not lams:
        lams = [(0.01, 0.01), (0.01, 0.01), (0.01, 0.01), (0.02, 0.005)]

    seed_strategies = ['random']
    degree_distributions = ['powerlaw', 'poisson']
    gamma = 2.9
    add_edges = None

    real_world_hg = None
    if real_world:
        sp_ms = [0.0, 1.0]
        homophilies = [real_world]

        seed_strategies = ['degree']
        try:
            with open(hg_file_path, 'rb') as f:
                hg = pkl.load(f)
            real_world_hg = hg
        except:
            temp_path = hg_file_path + '0.pkl'
            with open(temp_path, 'rb') as f:
                hg = pkl.load(f)
            real_world_hg = hg

        p_ms = [hg.n[1] / hg.N]
        degree_distributions = ['real']

    for p_m in p_ms:
       for sp_m in sp_ms:
            if sp_m=='prop':
                sp_m=p_m
            for h in homophilies:
                for d_s in degree_distributions:
                    for nu, lam in zip(nus, lams):
                            for seed_strategy in seed_strategies:
                                try:
                                    print(f'Generating params for p_m={p_m}, homophily={h}, nu={nu}, lam={lam}, seed_strategy={seed_strategy}, d_s={d_s}\n')

                                    # get edge counts
                                    if real_world:
                                        m = hg.m
                                        m = {key: {i: value for i, value in enumerate(values)} for key, values in m.items()}
                                    else:
                                        m = get_edge_counts(h = h, p_m = p_m)
                                    kbar = np.sum([k*np.sum(list(m[k].values())) for k in m.keys()])/NUM_NODES

                                    if add_edges is not None: # if you want to add edges
                                        for edge_size, num_edges in add_edges.items():
                                            m = get_edge_counts(h = h, p_m = p_m)
                                            for ii in range(3):
                                                if ii == 0:
                                                    write_params(
                                                        file_path = file_path,
                                                        p_m = p_m,
                                                        edge_counts = m,
                                                        nu = nu,
                                                        lam = lam,
                                                        seed_strategy = seed_strategy,
                                                        homophily = f'({h},{edge_size},{0})',
                                                        d_s = d_s,
                                                        kbar = kbar,
                                                        gamma = None if d_s == 'poisson' else gamma,
                                                        sp_m = sp_m,
                                                        real_world_hg=real_world_hg,
                                                        real_pred_gender=real_pred_gender
                                                    )
                                                    continue

                                                m = add_het_edges(m, edge_size, int(num_edges))

                                                write_params(
                                                    file_path = file_path,
                                                    p_m = p_m,
                                                    edge_counts = m,
                                                    nu = nu,
                                                    lam = lam,
                                                    seed_strategy = seed_strategy,
                                                    homophily = f'({h},{edge_size},{ii*num_edges})',
                                                    d_s = d_s,
                                                    kbar = kbar,
                                                    gamma = None if d_s == 'poisson' else gamma,
                                                    sp_m = sp_m,
                                                    real_world_hg=real_world_hg,
                                                    real_pred_gender=real_pred_gender
                                                )
                                    else: # if you don't want to add heterophilous edges
                                        write_params(
                                            file_path = file_path,
                                            p_m = p_m,
                                            edge_counts = m,
                                            nu = nu,
                                            lam = lam,
                                            seed_strategy = seed_strategy,
                                            homophily = f'({h},{0},{0})',
                                            d_s = d_s,
                                            kbar = kbar,
                                            gamma = None if d_s == 'poisson' else gamma,
                                            sp_m = sp_m,
                                            real_world_hg=real_world_hg,
                                            hg_file_path=hg_file_path,
                                            real_pred_gender=real_pred_gender
                                        )


                                except Exception as e:
                                    print(f'Error: {e}\n')


def add_het_edges(edge_counts:Dict[int, Dict[int, int]], edge_size:Union[int, str, List], num_edges:int) -> Dict[int, Dict[int, int]]:
    """
    Add heterophilous edges to the edge counts for a given edge size, and update the homophily value.
    Parameters:
        edge_counts: dict of dict of int, edge counts for the hypergraph
        edge_size: int, string or list, size of the edges to add
        num_edges: int, number of edges to add
    Returns:
        edge_counts: dict of dict of int, updated edge counts
    """
    if edge_size == 'all':
        edge_size = list(edge_counts.keys())
    elif isinstance(edge_size, int):
        edge_size = [edge_size]

    # get the number of edges in each group
    for k in edge_size:
        divisor = k - 1 # number of heterophilous edges to split the num_edges between
        for t in edge_counts[k].keys():
            if t == 0 or t == k: # skip the edges that are already homophilous
                continue
            # add the heterophilous edges
            edge_counts[k][t] += num_edges // divisor

    return edge_counts



def get_edge_counts(h:str, p_m:float) -> Dict[int, Dict[int, int]]:
    """
    Get the edge counts for the hypergraph. Currently this is a manual selection for a given homophily and p_m that we have precalculated
    Parameters:
        n: list of int, number of nodes in each group
        hs: dict of arrays of shape (2, k) s.t. h[k][g,t] is the (t+1)-type homophily of group g nodes w.r.t. to size k edges.
        mtot: dict of int, total number of edges in each size of edge
        exact_baseline: bool, whether to use the exact baseline or not
        alternative: bool, whether to use the alternative homophily scores or not
    Returns:
        edge_counts: dict of dict of int, edge counts for the hypergraph
    """
    # balanced
    if p_m == 0.5:
        if h == 'neutral':
            m = {2: {0: 6250, 1: 12500, 2: 6250}, 3: {0: 1500, 1: 4500, 2: 4500, 3: 1500}, 4: {0: 375, 1: 1500, 2: 2250, 3: 1500, 4: 375}}
        elif h=='weak':
            m = {2: {0: 8334, 1: 8332, 2: 8334}, 3: {0: 2122, 1: 3878, 2: 3878, 3: 2122}, 4: {0: 546, 1: 1403, 2: 2102, 3: 1403, 4: 546}}     # strength 1.5
        elif h=='het':
            m = {2: {0: 3575, 1: 17850, 2: 3575}, 3: {0: 800, 1: 5200, 2: 5200, 3: 800}, 4: {0: 195, 1: 1604, 2: 2404, 3: 1604, 4: 195}}      # 0.5
        elif h=='weakneutral3':
            m = {2: {0: 8334, 1: 8332, 2: 8334}, 3: {0: 1500, 1: 4500, 2: 4500, 3: 1500}, 4: {0: 375, 1: 1500, 2: 2250, 3: 1500, 4: 375}}   # these are the same as above but add 3 edges that mirror what the 4 edges are doing    
        elif h=='neutralweak3':
            m = {2: {0: 6255, 1: 12500, 2: 6245}, 3: {0: 2122, 1: 3878, 2: 3878, 3: 2122}, 4: {0: 546, 1: 1403, 2: 2102, 3: 1403, 4: 546}}  # these are the same as above but add 3 edges that mirror what the 4 edges are doing
        elif h=='hetneutral3':
            m = {2: {0: 3575, 1: 17850, 2: 3575}, 3: {0: 1500, 1: 4500, 2: 4500, 3: 1500}, 4: {0: 375, 1: 1500, 2: 2250, 3: 1500, 4: 375}}   # these are the same as above but add 3 edges that mirror what the 4 edges are doing
        elif h=='neutralhet3':
            m = {2: {0: 6255, 1: 12500, 2: 6245}, 3: {0: 800, 1: 5200, 2: 5200, 3: 800}, 4: {0: 193, 1: 1604, 2: 2406, 3: 1604, 4: 193}}     # these are the same as above but add 3 edges that mirror what the 4 edges are doing
        else:
            raise ValueError('Invalid homophily value')
    
    # imbalanced	
    elif p_m== 0.25:
        if h=='neutral':
            m = {2: {0: 14062, 1: 9375, 2: 1563}, 3: {0: 5062, 1: 5063, 2: 1687, 3: 188}, 4: {0: 1898, 1: 2533, 2: 1266, 3: 280, 4: 23}}
        elif h=='weak':
            m = {2: {0: 21919, 1: 2421, 2: 660}, 3: {0: 7626, 1: 3144, 2: 1052, 3: 178}, 4: {0: 2850, 1: 1938, 2: 970, 3: 215, 4: 27}}     # strength 1.5
        elif h=='het':
            m = {2: {0: 7200, 1: 16520, 2: 1280}, 3:{0: 2570, 1: 6976, 2: 2326, 3: 128}, 4: {0: 951, 1: 3126, 2: 1561, 3: 348, 4: 14}}      # strength 0.5
        elif h=='weakneutral3':
            m = {2: {0: 21919, 1: 2421, 2: 660}, 3: {0: 5062, 1: 5063, 2: 1687, 3: 188},  4: {0: 1898, 1: 2533, 2: 1266, 3: 280, 4: 23}}  # these are the same as the above but add 3 edges that mirror the 4 edges.    
        elif h=='neutralweak3':
            m = {2: {0: 14062, 1: 9375, 2: 1563}, 3: {0: 7626, 1: 3144, 2: 1052, 3: 178}, 4: {0: 2850, 1: 1938, 2: 970, 3: 215, 4: 27}}   # these are the same as the above but add 3 edges that mirror the 4 edges.    
        elif h=='hetneutral3':
            m = {2: {0: 7200, 1: 16520, 2: 1280}, 3: {0: 5062, 1: 5063, 2: 1687, 3: 188},  4: {0: 1898, 1: 2533, 2: 1266, 3: 280, 4: 23}} # these are the same as the above but add 3 edges that mirror the 4 edges.
        elif h=='neutralhet3':
            m = {2: {0: 14062, 1: 9375, 2: 1563}, 3:{0: 2570, 1: 6976, 2: 2326, 3: 128}, 4: {0: 951, 1: 3126, 2: 1561, 3: 348, 4: 14}} # these are the same as the above but add 3 edges that mirror the 4 edges.
        else:
            raise ValueError('Invalid homophily value')
    else:
        raise ValueError('Invalid p_m value')
    
    return m


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
    print('reading params')
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
    print('done w/ params, about to make edge_counts')
    # make all the edge_counts keys ints and the keys of the values ints
    if not real_world:
        edge_counts = {int(k): {int(kk): v for kk, v in v.items()} for k, v in edge_counts.items()}
    print('done with edge_counts, about to make n0 n1')
    n1 = int(num_nodes * p_m)
    n0 = num_nodes - n1
    print('done with n0 n1')
    # set up dictionary for seed generation time
    time_data = {'fp': [], 'hg': [], 'degree': [], 'hyper_core': [], 'k_core': [], 'random': []}

    # run for a num_hg hypergraphs
    for i in tqdm.tqdm(range(num_hg)):
        print('inside for loop')
        if real_world:
            if real_pred_gender:
                path_to_hg = original_path + str(i) + '.pkl'
            else:
                path_to_hg = original_path
        print(f'Running hypergraph {i}')

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
                print('getting hypergraph')
                edges, group, kappa = get_hypergraph(file_path, i, path_to_hg)
                print('got hypergraph')
            else:
                edges, group, kappa = get_hypergraph(file_path, i)
        else: # generate hypergraph
            edges, group, kappa = CSCM([n0, n1], edge_counts, degree_sampler, params=params, seed=hg_seed*i)
        
        if hypergraphs_only: # save only the hypergraph
            with open(f'{file_path}hg{i}.pkl', 'wb') as f:
                pkl.dump((edges, group, kappa), f)
            continue

        print('creating hASI object')
        hASI = HyperGraphAsymmetricSI(nodes=np.arange(0, num_nodes), edges=edges, group=group, nus=np.array([[nu[0],nu[1]],[nu[1],nu[0]]]), lams=np.array([[lam[0],lam[1]],[lam[1],lam[0]]]), fast=FAST, samplable_set=SAMPLABLE_SET)
        print('created hASI object')
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
                
                # print(seed_strategy)
                time_data[seed_strategy].append(elapsed)
                print('about to run simulation')
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


def run_all(file_path='./', hypergraphs_only=False, read_hypergraph=False, include_pairwise=False, seed_test=False, real_world=False, real_pred_gender=False, nu=None):
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
    if nu:
        lam_vals = [0.0825, 0.085, 0.0875, 0.09, 0.0925, 0.095, 0.0975, 0.10]
        # dirs = [d for d in dirs if os.path.isdir(f'{file_path}{d}') and '__pycache__' not in d and f'nu({nu},' in d]
        directs = []
        for y in lam_vals:
            directs.extend([d for d in dirs if os.path.isdir(
                f'{file_path}{d}') and '__pycache__' not in d and f'nu({nu},' in d and f'lam(0.01, {y}' in d])
            # dirs = [d for d in dirs if os.path.isdir(f'{file_path}{d}') and '__pycache__' not in d and f'nu({nu},' in d and f'lam(0.01, {y}' in d]
        dirs = directs
    else:
        dirs = os.listdir(file_path)

    for dir in tqdm.tqdm(dirs):
        if os.path.isdir(dir) and not 'pycache' in dir:
            print(f'Running simulations for {dir}')
            file_path_dir = f'{file_path}{dir}/'
            params = get_params(file_path_dir)
            run(params, file_path_dir, hypergraphs_only=hypergraphs_only, read_hypergraph=read_hypergraph, include_pairwise=include_pairwise, seed_test=seed_test, real_world=real_world, real_pred_gender=real_pred_gender)

def main():

    run_all(file_path= './', hypergraphs_only=False, read_hypergraph=False, include_pairwise=False, seed_test=False)

if __name__=='__main__':
    main()

