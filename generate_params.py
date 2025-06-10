"""
Generate Params

Generate a specified json file with parameter values to generate a hypergraph (or access a real-world one) and run a
set of simulations and output. Throughout this file, Group 0 is the majority group and Group 1 is the minority group.


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

# IMPORTS #
import json
import numpy as np
import os
import pandas as pd
import pickle as pkl
import time
import tqdm
from typing import Dict, Tuple, List, Set

from HyperGraph import * 
from HyperGraphHelper import *

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

    homophilies = ['weak', 'neutral', 'het', 'neutralweak3', 'weakneutral3', 'hetneutral3', 'neutralhet3']
    if not nus:
        if real_world:
            if real_world == 'housebills' or real_world == 'senatebills':
                nus = [(1.0, 1.0), (1.33, 1.33), (0.75, 0.75), (1.33, 0.75)]
            else:
                nus = [(1.0, 1.0), (1.11, 1.11), (0.9, 0.9), (1.11, 0.9)]
        else:
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


def get_edge_counts(h: str, p_m: float) -> Dict[int, Dict[int, int]]:
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
            m = {2: {0: 6250, 1: 12500, 2: 6250}, 3: {0: 1500, 1: 4500, 2: 4500, 3: 1500},
                 4: {0: 375, 1: 1500, 2: 2250, 3: 1500, 4: 375}}
        elif h == 'weak':
            m = {2: {0: 8334, 1: 8332, 2: 8334}, 3: {0: 2122, 1: 3878, 2: 3878, 3: 2122},
                 4: {0: 546, 1: 1403, 2: 2102, 3: 1403, 4: 546}}  # strength 1.5
        elif h == 'het':
            m = {2: {0: 3575, 1: 17850, 2: 3575}, 3: {0: 800, 1: 5200, 2: 5200, 3: 800},
                 4: {0: 195, 1: 1604, 2: 2404, 3: 1604, 4: 195}}  # 0.5
        elif h == 'weakneutral3':
            m = {2: {0: 8334, 1: 8332, 2: 8334}, 3: {0: 1500, 1: 4500, 2: 4500, 3: 1500},
                 4: {0: 375, 1: 1500, 2: 2250, 3: 1500,
                     4: 375}}  # these are the same as above but add 3 edges that mirror what the 4 edges are doing
        elif h == 'neutralweak3':
            m = {2: {0: 6255, 1: 12500, 2: 6245}, 3: {0: 2122, 1: 3878, 2: 3878, 3: 2122},
                 4: {0: 546, 1: 1403, 2: 2102, 3: 1403,
                     4: 546}}  # these are the same as above but add 3 edges that mirror what the 4 edges are doing
        elif h == 'hetneutral3':
            m = {2: {0: 3575, 1: 17850, 2: 3575}, 3: {0: 1500, 1: 4500, 2: 4500, 3: 1500},
                 4: {0: 375, 1: 1500, 2: 2250, 3: 1500,
                     4: 375}}  # these are the same as above but add 3 edges that mirror what the 4 edges are doing
        elif h == 'neutralhet3':
            m = {2: {0: 6255, 1: 12500, 2: 6245}, 3: {0: 800, 1: 5200, 2: 5200, 3: 800},
                 4: {0: 193, 1: 1604, 2: 2406, 3: 1604,
                     4: 193}}  # these are the same as above but add 3 edges that mirror what the 4 edges are doing
        else:
            raise ValueError('Invalid homophily value')

    # imbalanced
    elif p_m == 0.25:
        if h == 'neutral':
            m = {2: {0: 14062, 1: 9375, 2: 1563}, 3: {0: 5062, 1: 5063, 2: 1687, 3: 188},
                 4: {0: 1898, 1: 2533, 2: 1266, 3: 280, 4: 23}}
        elif h == 'weak':
            m = {2: {0: 21919, 1: 2421, 2: 660}, 3: {0: 7626, 1: 3144, 2: 1052, 3: 178},
                 4: {0: 2850, 1: 1938, 2: 970, 3: 215, 4: 27}}  # strength 1.5
        elif h == 'het':
            m = {2: {0: 7200, 1: 16520, 2: 1280}, 3: {0: 2570, 1: 6976, 2: 2326, 3: 128},
                 4: {0: 951, 1: 3126, 2: 1561, 3: 348, 4: 14}}  # strength 0.5
        elif h == 'weakneutral3':
            m = {2: {0: 21919, 1: 2421, 2: 660}, 3: {0: 5062, 1: 5063, 2: 1687, 3: 188},
                 4: {0: 1898, 1: 2533, 2: 1266, 3: 280,
                     4: 23}}  # these are the same as the above but add 3 edges that mirror the 4 edges.
        elif h == 'neutralweak3':
            m = {2: {0: 14062, 1: 9375, 2: 1563}, 3: {0: 7626, 1: 3144, 2: 1052, 3: 178},
                 4: {0: 2850, 1: 1938, 2: 970, 3: 215,
                     4: 27}}  # these are the same as the above but add 3 edges that mirror the 4 edges.
        elif h == 'hetneutral3':
            m = {2: {0: 7200, 1: 16520, 2: 1280}, 3: {0: 5062, 1: 5063, 2: 1687, 3: 188},
                 4: {0: 1898, 1: 2533, 2: 1266, 3: 280,
                     4: 23}}  # these are the same as the above but add 3 edges that mirror the 4 edges.
        elif h == 'neutralhet3':
            m = {2: {0: 14062, 1: 9375, 2: 1563}, 3: {0: 2570, 1: 6976, 2: 2326, 3: 128},
                 4: {0: 951, 1: 3126, 2: 1561, 3: 348,
                     4: 14}}  # these are the same as the above but add 3 edges that mirror the 4 edges.
        else:
            raise ValueError('Invalid homophily value')
    else:
        raise ValueError('Invalid p_m value')

    return m


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


def main():

    generate_params()

if __name__=='__main__':
    main()

