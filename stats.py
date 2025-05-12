"""
Stats

Extract and store descriptive statistics from hypergraphs.

ML - 2025/05/12
"""

### IMPORT ###
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pickle

from HyperGraph import *
from HyperGraphHelper import *

### PARAMETERS ###

input_path = './hypergraphs/'
output_path = './stats/'
edge_sizes = [2,3,4,5]
groups = [0, 1]
alternative = True
verbose = True

sub_dir = {
    'aps_genderapi' : 'aps/hypergraphs_genderapi/',
    'aps_genderizerio' : 'aps/hypergraphs_genderizerio/',
    'dblp_genderapi' : 'dblp/hypergraphs_genderapi/',
    'dblp_genderizerio' : 'dblp/hypergraphs_genderizerio/',
    'highschool' : 'highschool/hypergraphs/',
    'primaryschool' : 'primaryschool/hypergraphs/',
    'hospital' : 'hospital/hypergraphs/',
    'housebills' : 'housebills/hypergraphs/',
    'senatebills' : 'senatebills/hypergraphs/',
    'housebillsgender_genderapi' : 'housebillsgender/hypergraphs_genderapi/',
    'housebillsgender_genderizerio' : 'housebillsgender/hypergraphs_genderizerio/',
    'senatebillsgender_genderapi' : 'senatebillsgender/hypergraphs_genderapi/',
    'senatebillsgender_genderizerio' : 'senatebillsgender/hypergraphs_genderizerio/'
}

### MAIN ###
for counter, hypergraph in enumerate(list(sub_dir.keys())):

    if verbose: print(f'{counter}{len(list(sub_dir.keys()))}, Processing: {hypergraph}.')

    # list all hypergraphs and determine how many there are
    hg_files = os.listdir(f'{input_path}{sub_dir[hypergraph]}')
    n_hg = len(hg_files)

    # initialize datastructures to store results
    homophily_dict = {g: {s: np.zeros((n_hg, s+1)) for s in edge_sizes} for g in groups}
    num_nodes = np.zeros((n_hg, len(groups)+1))
    num_edges = np.zeros(len(edge_sizes))
    k_bar = np.zeros((n_hg, len(groups)+1))
    k_sqr = np.zeros((n_hg, len(groups)+1))
    smax = 0.
    mtot = 0.

    # process the hypergraphs in the hypergraph directory
    for i, hg_file in enumerate(hg_files):

        # print progress
        if verbose: print(f"{i+1}/{n_hg}", end="\r")
    
        # open File
        with open(f"{input_path}{sub_dir[hypergraph]}{hg_file}", 'rb') as f:
    
            hg = pickle.load(f)
    
        # store the number of nodes
        num_nodes[i, :] = hg.n + [np.sum(hg.n)]
    
        # store the number of edges
        if i == 0:
            num_edges = np.array([np.sum(list(hg.m[s].values())) for s in edge_sizes])
            mtot = np.sum([np.sum(list(hg.m[s].values())) for s in hg.m.keys()])
            smax = np.max(list(hg.m.keys()))
    
        # degrees
        k = np.array([hg.degree[v] for v in hg.nodes])
        group = np.asarray(hg.group)
    
        # store the average degree
        k_bar[i, 0] = np.mean(k[group == 0])
        k_bar[i, 1] = np.mean(k[group == 1])
        k_bar[i, 2] = np.mean(k)
        
        # store the second moment of the degree distribution
        k2 = np.power(k, 2)
        k_sqr[i, 0] = np.mean(k2[group == 0])
        k_sqr[i, 1] = np.mean(k2[group == 1])
        k_sqr[i, 2] = np.mean(k2)
    
        # Calculate Hypergraph Homophily
        h = hg.homophily(alternative=alternative)
    
        # Store the Hypergraph Homophily
        for g in groups:
            for s in edge_sizes:
                for num_vg in range(1, s+1):
        
                    if g == 0:
                        r = s - num_vg
                    elif g == 1:
                        r = num_vg
                    else:
                        continue
                    
                    homophily_dict[g][s][i, r] = h[g][s].get(num_vg, np.nan)

    # collect and store data
    output_dict = {
        'num_nodes' : num_nodes,
        'num_edges' : num_edges,
        'smax'      : smax,
        'k_bar'     : k_bar,
        'k_sqr'     : k_sqr,
        'homophily' : homophily_dict,
        'mtot'      : mtot
    }

    # store the homophily values
    with open(f'{output_path}{hypergraph}.pkl', "wb") as f:
    
        pickle.dump(output_dict, f)