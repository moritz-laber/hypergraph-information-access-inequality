"""
Prepare Data Real World

Preprocesses the data from simulations on 
synthetic hypergraphs to create plots.

ML - 2025/05/12
"""

### IMPORT ###
import numpy as np
import pickle

from plot_functions import prepare_data_synthetic

### PARAMETERS ###

base_dir = '/simulations/'          # input directory
output_file = './results.pkl'       # output file name

num_nodes = 10_000                   # number of nodes
pm = 0.25                            # proportion of minority nodes
spms = [0.0, pm, 1.0]                # proportion of minority seeds
degree_distribution = 'poisson'      # degree distribution
seeding_strategy = 'random'          # seeding strategy

homophily = [
    'neutral', 'weak', 'het',
    'neutralweak3', 'weakneutral3', 'hetneutral3',
    'neutralhet3', 'hetweak3', 'weakhet3'
]                  

dynamics = [('linear',      (1.0, 1.0), (0.01, 0.01)),  # dynamics
            ('sublinear',   (0.5, 0.5), (0.01, 0.01)),
            ('superlinear', (2.0, 2.0), (0.01, 0.01)),
            ('asymmetric',  (2.0, 0.5), (0.02, 0.005))]

fmin = 0.00  # fraction of nodes minimum value
fmax = 0.90  # fraction of nodes maximum value
fnum = 200   # number of values between minimum and maximum (inclusive)

verbose = True

### MAIN ###

if __name__ == "__main__":

    # compute the fraction of nodes values at 
    # which to store the outcome of the information
    # contagion process
    fs = np.linspace(fmin, fmax, fnum)

    # data structure to store the results
    result = {}

    # load the data selected according to the parameters
    for spm in spms:
        for dyn, nu, lam in dynamics:
            for hom in homophily:
                
                directory = f'{base_dir}pm{pm}_spm{spm}_h({hom},0,0)_ds{degree_distribution}_nu{nu}_lam{lam}_ss{seeding_strategy}/'    
                if verbose: print(f"loading: {directory}")
                result[(spm, dyn, hom)] = prepare_data_synthetic(directory, fs)

    # save the results to file
    output_dict = {
     'result' : result,
     'params' : {
         'num_nodes' : num_nodes,
         'pm'       : pm,
         'spms'     : spms,
         'degree_distribution' : degree_distribution,
         'seeding_strategy' : seeding_strategy,
         'homophily' : homophily,
         'dynamics'  : dynamics,
         'fs'       : fs
     }
    }

    # with pickle save
    with open(output_file, 'wb') as f:

        pickle.dump(output_dict, f)


    if verbose: print('Done.')



    
    