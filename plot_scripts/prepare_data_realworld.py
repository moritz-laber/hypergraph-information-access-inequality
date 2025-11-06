"""
Prepare Data Real-World

Preprocesses the data from simulations on 
real-world hypergraphs to create plots.

ML - 2025/05/12
"""

### IMPORT ###
import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import os
import pandas as pd
import pickle
import re
from scipy.stats import wasserstein_distance, gaussian_kde
from tqdm import tqdm

from simulation import get_params

## PARAMETERS ##
fmin = 0.           # minimum fraction of informed nodes
fmax = 0.90         # maximum fraction of informed nodes
nf = 200            # number of fractions of informed nodes to consider

degree_distribution = 'real'     # degree distribution
seeding_strategy = 'random'      # seeding strategy

hypergraphs = ['aps',
               'dblp',
               'housebillsgender', 
               'housebills',
               'senatebillsgender',
               'senatebills',
               'hospital',
               'primaryschool',
               'highschool'
                ]

basedir = './simulations/'  # input directory
api = 'genderapi'             

subdir = {hg :  f'{hg}/simulations_{api}' if hg in ['aps', 'dblp', 'housebillsgender', 'senatebillsgender'] else f'{hg}/simulations' for hg in hypergraphs}

outdir = './results/'       # output directory

pm = 'real'

spms = [0.0, 1.0]

dynamics = ['linear',
            'sublinear',
            'superlinear',
            'asymmetric']

num_processes = 8

## CONSTANTS ##

TIME_COLUMN = 1
GROUP_COLUMN = 2
RANK_COLUMN = -1
EDGE_SIZE_COLUMN = 4

GROUP_LIST = [0, 1]
EDGE_SIZE_LIST = [2,3,4]

## FUNCTION DEFINITIONS ##

def prepare_data(input_dict):

    inputdir = input_dict['inputdir']
    fs = input_dict['fs']
    num_hg = input_dict['num_hg']

    params = get_params(inputdir)
    num_seed_cond = params['num_seed_cond']
    seed_strategy = params['seed_strategy']
    num_nodes = params['num_nodes']
    pm = params['p_m']
    if not num_hg:
        num_hg = params['num_hg']
    
    emd = []
    tf = []
    tfmax = {0: [], 1:[]}
    i1f = []
    edge_sizes = {s: [[], []] for s in EDGE_SIZE_LIST}
    
    num_timeseries = 0

    # calculate the indices at which a fraction f of all nodes is infected
    if fs.shape[0] > num_nodes:
        print("to few node for given fs resolution falling back to 1/n \n")
        fs = np.linspace(fs.min(), fs.max(), int(num_nodes))

    fmax = np.max(fs)
    f_idx = np.floor(fs * num_nodes).astype(np.int32)

    iteration = 0
    failed = []
    for hg_num in range(num_hg):
        for seeding_num in range(num_seed_cond):
            
            # load one simulation output file
            try:
                file_path = f'{inputdir}results_hg{hg_num}_{seed_strategy}{seeding_num}.pkl'
                results = pd.read_pickle(file_path)
            except:
                failed.append((iteration, file_path))
                continue
                

            # aggregate the results from different runs
            for run_num, result in results.items():
                iteration +=1
                
                # truncate the time series s.t. the maximum number of infected nodes
                # is the same in all runs. This makes sure that the averages are 
                # taken over the same number of samples at all values of f.
                group = result[:, GROUP_COLUMN]

                i0_max = (group==0).shape[0]
                i1_max = (group==1).shape[0]
                i_max = group.shape[0]

                ig_cutoff = (int(fmax * num_nodes * (1. - pm)), int(fmax * num_nodes * pm))
                i_cutoff = int(fmax * num_nodes)

                if ig_cutoff[0] > i0_max or ig_cutoff[1] > i1_max or i_cutoff > i_max:
                    continue
                else:

                    # increasse the counter of eligible time series
                    num_timeseries += 1
                    
                    # rank the nodes by time of infection
                    rank_column = np.arange(i_max).reshape(-1, 1)
                    ranked_result = np.hstack((result, rank_column))

                    # extract the group membership of nodes
                    group = ranked_result[:, GROUP_COLUMN]

                    # determine emd for each run
                    emd.append(wasserstein_distance(ranked_result[group==0, RANK_COLUMN][:ig_cutoff[0]], ranked_result[group==1, RANK_COLUMN][:ig_cutoff[1]]))

                    # determine the times tf at which fraction f of all nodes is informed
                    tf.append(ranked_result[f_idx, TIME_COLUMN])
                    
                    # determine how many minority nodes are infected at times tf
                    i1f.append(np.cumsum(ranked_result[:, GROUP_COLUMN], dtype=np.float32)[f_idx]/(pm * num_nodes))

                    # determine the times at which a fraction fmax of minority or majority nodes is informed
                    idx1 = np.argmin(np.abs(np.cumsum(group) - ig_cutoff[1]))
                    idx0 = np.argmin(np.abs(-np.cumsum(group - 1) - ig_cutoff[0]))

                    tfmax[0].append(ranked_result[idx0, TIME_COLUMN])
                    tfmax[1].append(ranked_result[idx1, TIME_COLUMN])

                    # determine the sizes of edges present in infection events
                    for s in EDGE_SIZE_LIST:
                        for g in GROUP_LIST:
                            edge_sizes[s][g].append(np.sum(ranked_result[group==g, EDGE_SIZE_COLUMN]==s))
    
    output = {}
    output['id'] = inputdict['id']
    output['num_timeseries'] = num_timeseries
    output['emd'] = emd
    output['edge_sizes'] = edge_sizes
    output['fs'] = fs
    output['tfs'] = tf
    output['tfmax'] = tfmax
    output['i1f'] = i1f
    output['failed'] = failed

    return output

if __name__ == "__main__":

    # the values for the fraction of nodes at which to
    # evaluate the metrics.
    fs = np.linspace(fmin, fmax, num=nf)
    
    counter = 0
    for hg in hypergraphs:

        inputlist = []
        keylist = []

        if hg in ['aps', 'dblp', 'housebillsgender', 'senatebillsgender']:
            num_hg = 1000
            outputfile = f'{outdir}/{hg}_{api}_plotdata.pkl'
        else:
            num_hg = None
            outputfile = f'{outdir}/{hg}_plotdata.pkl'
        
        for dyn in dynamics:
            for spm in spms:
                
                counter += 1
                if hg in ['aps','dblp','housebillsgender','senatebillsgender','primaryschool','highschool','hospital']:
                    if dyn == 'sublinear':
                        nu = (0.9,0.9)
                        lam = (0.01,0.01)
                    elif dyn == 'superlinear':
                        nu = (1.11,1.11)
                        lam = (0.01,0.01)
                    elif dyn == 'linear':
                        nu = (1.0,1.0)
                        lam = (0.01,0.01)
                    elif dyn == 'asymmetric':
                        nu = (1.11,0.9)
                        lam = (0.02,0.005)
                elif hg in ['housebills','senatebills']:
                    if dyn == 'sublinear':
                        nu = (0.75,0.75)
                        lam = (0.01,0.01)
                    elif dyn == 'superlinear':
                        nu = (1.33,1.33)
                        lam = (0.01,0.01)
                    elif dyn == 'linear':
                        nu = (1.0,1.0)
                        lam = (0.01,0.01)
                    elif dyn == 'asymmetric':
                        nu = (1.33,0.75)
                        lam = (0.02,0.005)
                else:
                    print(hg)
    
    
                inputdir = f'{basedir}/{subdir[hg]}/pm{pm}_spm{spm}_h({hg},0,0)_ds{degree_distribution}_nu{nu}_lam{lam}_ss{seeding_strategy}/'

                inputdict = {
                    'inputdir' : inputdir,
                    'fs' : fs,
                    'num_hg' : num_hg,
                    'id': (spm, dyn, hg)
                }

                keylist.append((spm, dyn, hg))
                inputlist.append(inputdict)

        # process the data in parallel
        num_processes = mp.cpu_count()
        print(f'processes: {num_processes}')
        with mp.Pool(processes=num_processes) as pool:
            outputlist = pool.map(prepare_data, inputlist)

        # create dictionary from results
        result = dict(zip(keylist, outputlist))

        # write dictionary to file
        with open(outputfile, 'wb') as f:
        
            pickle.dump(result, f)