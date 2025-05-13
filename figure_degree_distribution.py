"""
Figure & Simulation: Degree Distribution

This script generates plot of the degree
distribution, and expected degree distribution.

You can use degree sequences saved to file,
or generate new graphs.

ML - 2025/05/13
"""

### IMPORTS ###
from HyperGraph import *
from simulation import get_edge_counts

import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pickle

### PARAMETERS ###

# hypergraph parameters
n_hg = 1000                                # number of hypergraphs to average over
n = 10_000                                 # number of nodes
gamma = 2.9                                # exponent of the expected degree distribution
pms = [0.5, 0.25]                          # fractions of group 1 nodes
group_labels = [0, 1]                      # groups

degree_seed = 1245                         # seed for sampling expected degrees
hypergraph_seed = 1235                     # seed for hypergraph generator

new_hypergraphs = True                     # whether to generate new hypergraphs or read results from file
output_file = './degree_distribution.pkl'  # where to store new hypergraphs

verbose = True                             # whether to print progress

# plot parameters
input_file = output_file                    # from where to read results if new_hypergraphs is false
figure_name = './figure_degreedistribution' # where to save the plot to
dpi = 600                                   # dots per inch for png version

figsize = (7.2, 5.2)                        # size of the figure in inches
nrows = 3                                   # number of gridspec rows
ncols = 3                                   # number of gridspec columns
wratios = [1., 1., 1.]                      # ratio of widths of gridspec cells
hratios = [1., 1., 0.05]                    # ratio of heights of gridspec cells
wspace = 0.20                               # horizontal separation in gridspec
hspace = 0.60                               # verticle separation in gridspec
color_kappa = ('#c2d58a','#6d71b7')         # colors for groups (0,1) for expected degree
color_degree = ('#a6b96f', '#53589c')       # colors for groups (0,1) for realized degree
marker_kappa = {0:'s', 1 : 'o'}             # markers for groups (0,1) for expected degee
marker_degree = {0:'x', 1: '+'}             # markers for groups (0,1) for realized degree
markersize_kappa = {0:7, 1:6}               # marker sizes for groups (0,1) for expected degree
markersize_degree = {0:6, 1:7}              # marker sizes for groups (0,1) for realized degree
alpha = 0.65                                # transparency for expected degree

xmin = 1.0                                  # minimum value for binning
xmax = None                                 # maximum value. If None then set data dependent
nbins = 15                                  # number of bins
 
xlim = (9.5e-1, 2e4)                        # x-axis limits 
ylim = (7.5e-8, 1.1e0)                        # y-axis limits

yticks = [1e-2, 1e-4, 1e-6]
yticklabels = [r'$10^{-2}$', r'$10^{-4}$', r'$10^{-6}$']

labelsize = 12                              # size of axis labels
ticksize = 10                               # size of tickmarkers
textsize = 12                               # size of subplotlabels
textfontweight = 'bold'                     # fontweight of subplotlabels
textcoords = (-0.15, 1.05)                  # coordinates of subplotlabels
letters = {(0,0) : 'a',                     # subplotlabels
           (0,1) : 'b', 
           (0,2) : 'c',
           (1,0) : 'd',
           (1,1) : 'e',
           (1,2) : 'f'}
headercoords = (0.29, 1.05)                   # coordinates for subplot headers
headersize = 12                               # size of subplotheaders
to_header = {('weak', 0.5)    : 'hom. bal.',  # subplotheaders
             ('weak', 0.25)   : 'hom. imb.',
             ('het', 0.5)     : 'het. bal.',
             ('het', 0.25)    : 'het. imb.',
             ('neutral', 0.5) : 'neu. bal.',
             ('neutral', 0.25): 'neu. imb.'}
ncol_legend = 4                              # number of legend columsn
pos_legend = (0.5, -5)                       # legend position
fontsize_legend = 10                         # legend fontsize

### MAIN ###

# Hypergraph generation

if new_hypergraphs:
    results = {}
    for i, pm in enumerate(pms):
        for j, homophily in enumerate(['weak', 'neutral', 'het']):
    
            # determine the number of nodes in each group
            n0 = int(n * (1.0 - pm))
            n1 = int(n * pm)
    
            # determine the edge counts
            m = get_edge_counts(homophily, pm)
    
            kappas = {g : [] for g in group_labels}
            degrees = {g : [] for g in group_labels}
            
            for ii in range(n_hg):
                
                if verbose: print(f'homophily: {homophily}, pm={pm}, num: {ii+1}/{n_hg}', end='\r')
    
                # create sampler for hidden degrees
                degree_seed += 24
                degree_rng = np.random.default_rng(seed=degree_seed)
                degree_sampler = lambda n, kbar, gamma : ((gamma - 2.)/(gamma - 1.))*kbar*(1 + degree_rng.pareto(gamma - 1, size=n))
                params = (gamma,)
    
                hypergraph_seed += 25
                edges, group, kappa = CSCM([n0,n1], m, degree_sampler, params=params, seed=hypergraph_seed)
                H = HyperGraph(nodes=np.arange(0, n0+n1), edges=edges, group=group)
            
                for g in group_labels:
                    kappas[g] += list(kappa[g])
                    degrees[g] += list([H.degree[i] for i in H.nodes if H.group[i]==g])
    
    
            # determine the bins
            if xmax == None:
                xmax = np.max(kappas[0] + kappas[1] + degrees[0] + degrees[1])
            bin_edges = np.logspace(np.log10(xmin), np.log10(xmax), num=nbins+1, base=10)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
            # bin observed and expected degrees
            kappas_binned = {}
            degrees_binned = {}
            for g in [0,1]:
                kappas_binned[g], _ = np.histogram(kappas[g], bins=bin_edges, density=True)
                degrees_binned[g], _ = np.histogram(degrees[g], bins=bin_edges, density=True)
    
            results[(homophily, pm)] = {'kappas' : kappas_binned,
                                        'degrees':degrees_binned,
                                        'bin_centers' : bin_centers}
    
            if verbose: print('\n')

    with open(output_file, 'wb') as f:

        pickle.dump(results, f)
        

else:

    with open(input_file, 'rb') as f:

        results = pickle.load(f)


# Plots
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(nrows, ncols, wspace=wspace, hspace=hspace, width_ratios=wratios, height_ratios=hratios, figure=fig)

for i, pm in enumerate(pms):
    for j, homophily in enumerate(['weak', 'neutral', 'het']):

        # unpack results
        bin_centers = results[(homophily, pm)]['bin_centers']
        kappas_binned = results[(homophily, pm)]['kappas']
        degrees_binned = results[(homophily, pm)]['degrees']
        
        # plot the results
        ax = fig.add_subplot(gs[i,j])
        for g in group_labels:
            ax.plot(bin_centers, kappas_binned[g], color=color_kappa[g], marker=marker_kappa[g], markersize=markersize_kappa[g], alpha=alpha, ls='')
            ax.plot(bin_centers, degrees_binned[g], color=color_degree[g], marker=marker_degree[g], markersize=markersize_degree[g], ls='')
        ax.set_xscale('log')
        ax.set_xlim(xlim)
        ax.set_yscale('log')
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        if j!=0:
            ax.yaxis.set_visible(False)

        ax.text(textcoords[0], textcoords[1], letters[(i,j)], fontsize=textsize, fontweight=textfontweight, transform=ax.transAxes)
        ax.text(headercoords[0], headercoords[1], to_header[(homophily,pm)], fontsize=headersize, transform=ax.transAxes)
        
        if i==1:
            ax.set_xlabel(r'(expected) degree', fontsize=labelsize)
        if j==0:
            ax.set_ylabel(r'density', fontsize=labelsize)

        ax.tick_params(labelsize=ticksize)

# add legend
handles = [
    Line2D([0], [0], marker=marker_kappa[0], color=color_kappa[0], linestyle='', markersize=markersize_kappa[0], label=r'$\rho_0(\kappa)$'),
    Line2D([0], [0], marker=marker_kappa[1], color=color_kappa[1], linestyle='', markersize=markersize_kappa[1], label=r'$\rho_1(\kappa)$'),
    Line2D([0], [0], marker=marker_degree[0], color=color_degree[0], linestyle='', markersize=markersize_degree[0], label=r'$p_0(k)$'),
    Line2D([0], [0], marker=marker_degree[1], color=color_degree[1], linestyle='', markersize=markersize_degree[1], label=r'$p_1(k)$'),
]

ax = fig.add_subplot(gs[-1, :])
ax.axis('off')
ax.legend(handles=handles, ncol=ncol_legend, loc='lower center', bbox_to_anchor=pos_legend, fontsize=fontsize_legend)

fig.savefig(f'{figure_name}.pdf')
fig.savefig(f'{figure_name}.svg')
fig.savefig(f'{figure_name}.png', dpi=dpi)