"""
PAIRWISE DEGREE DISTRIBUTIONS
"""

### IMPORT ###
import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import numpy as np
import pickle

from plot_functions import *

### PARAMETERS ###

# I/O Parameters
results_dir = './degreedistribution_realworld/'
output_path = './'
figname = 'fig_degreedistribution_pairwise'


pairwise = True

hypergraphs = [
    'primaryschool',
    'highschool',
    'hospital',
    'housebills',
    'housebillsgender_genderapi',
    'housebillsgender_genderizerio',
    'senatebills',
    'senatebillsgender_genderapi',
    'senatebillsgender_genderizerio',
    'aps_genderapi',
    'aps_genderizerio',
    'dblp_genderapi',
    'dblp_genderizerio',
]

hypergraph_settings = {
    'primaryschool' : {
        'name' : 'Primary School',
        'xlim' : (5, 200),
        'xticks' : [1e1, 1e2],
        'ylim' : (8e-4, 5e-2),
        'yticks' : [1e-2, 1e-3],
        'logbin' : True,
    },
    'highschool' : {
        'name' : 'High School',
        'xlim' : (0.5, 300),
        'xticks' : [1e0, 1e1, 1e2],
        'ylim' : (8e-4, 6e-2),
        'yticks' : [1e-2, 1e-3],
        'logbin' : True
    },
    'hospital' : {
        'name' : 'Hospital',
        'xlim' : (0, 77),
        'xticks' : [25, 50, 75],
        'ylim' : (0, 0.1),
        'yticks' : [0.04, 0.08],
        'logbin' : False
    },
    'housebills' : {
        'name' : 'House\n(Party)',
        'xlim' : (0.5, 7000),
        'xticks' : [1e0, 1e1, 1e2, 1e3],
        'ylim' : (1e-6, 1e-1),
        'yticks' : [1e-2, 1e-4],
        'logbin' : True
    },
    'housebillsgender_genderapi' : {
        'name' : 'House\n(GenderAPI)',
        'xlim' : (0.5, 7000),
        'xticks' : [1e0, 1e1, 1e2, 1e3],
        'ylim' : (8e-7, 1e-2),
        'yticks' : [1e-3, 1e-6],
        'logbin' : True
    },
    'housebillsgender_genderizerio' : {
        'name' : 'House\n(GenderizerIO)',
        'xlim' : (0.5, 7000),
        'xticks' : [1e0, 1e1, 1e2, 1e3],
        'ylim' : (8e-7, 1e-2),
        'yticks' : [1e-3, 1e-6],
        'logbin' : True
    },
    'senatebills' : {
        'name' : 'Senate\n(Party)',
        'xlim' : (50, 1100),
        'xticks' : [1e2, 1e3],
        'ylim' : (9e-5, 3e-2),
        'yticks' : [1e-2, 1e-4],
        'logbin' : True
    },
    'senatebillsgender_genderapi' : {
        'name' : 'Senate\n(GenderAPI)',
        'xlim' : (50, 1100),
        'xticks' : [1e2, 1e3],
        'ylim' : (9e-5, 3e-2),
        'yticks' : [1e-2, 1e-4],
        'logbin' : True
    },
    'senatebillsgender_genderizerio' : {
        'name' : 'Senate\n(GenderizerIO)',
        'xlim' : (50, 1100),
        'xticks' : [1e2, 1e3],
        'ylim' : (9e-5, 3e-2),
        'yticks' : [1e-2, 1e-4],
        'logbin' : True
    },
    'aps_genderapi' : {
        'name' : 'APS\n(GenderAPI)',
        'xlim' : (0.5, 5000),
        'xticks' : [1e0, 1e1, 1e2, 1e3],
        'ylim' : (1e-6, 1.2),
        'yticks' : [1e-1, 1e-3, 1e-5],
        'logbin' : True
    },
    'aps_genderizerio' : {
        'name' : 'APS\n(GenderizerIO)',
        'xlim' : (0.5, 5000),
        'xticks' : [1e0, 1e1, 1e2, 1e3],
        'ylim' : (1e-6, 1.2),
        'yticks' : [1e-1, 1e-3, 1e-5],
        'logbin' : True
    },
    'dblp_genderapi' : {
        'name' : 'DBLP\n(GenderAPI)',
        'xlim' : (0.5, 1100),
        'xticks' : [1e0, 1e1, 1e2, 1e3],
        'ylim' : (9e-8, 1.2),
        'yticks' : [1e-3, 1e-6],
        'logbin' : True
    },
    'dblp_genderizerio' : {
        'name' : 'DBLP\n(GenderizerIO)',
        'xlim' : (0.5, 1100),
        'xticks' : [1e0, 1e1, 1e2, 1e3],
        'ylim' : (9e-8, 1.2),
        'yticks' : [1e-3, 1e-6],
        'logbin' : True
    }
}

coordinates = {
    0 : (1,1,3),
    1 : (1,3,5),
    2 : (1,5,7),
    3 : (3,1,3),
    4 : (3,3,5),
    5 : (3,5,7),
    6 : (5,1,3),
    7 : (5,3,5),
    8 : (5,5,7),
    9 : (7,0,2),
    10 : (7,2,4),
    11 : (7,4,6),
    12 : (7,6,8)
}

abc = {
    (1,1,3) : "(a)",
    (1,3,5) : "(b)",
    (1,5,7) : "(c)",
    (3,1,3) : "(d)",
    (3,3,5) : "(e)",
    (3,5,7) : "(f)",
    (5,1,3) : "(g)",
    (5,3,5) : "(h)",
    (5,5,7) : "(i)",
    (7,0,2) : "(j)",
    (7,2,4) : "(k)",
    (7,4,6) : "(l)",
    (7,6,8) : "(m)"
}

annotation_fontsizes = [11, 9]
annotation_fontweights = ['bold', 'normal']
annotation_coords = [(-0.10, 1.05), (0.50, 1.05)]

annotation_colors = ['k', '#272c2d']
small_annotation_color = '#525758'
spm = 0.

text_style = {
    'fontsize' : 10,
    'fontcolor' : '#14191a',
    'fontweight' : 'bold'
}

text_coords = (-0.05, 0.10)
text_line = "Distribution of the number of unique neighbors in real world hypergraphs"

# Figure Parameters
figsize = (7.2, 8.0)   # inches
figure_hspace = 1.0
figure_wspace = 1.0
figure_ncols = 8
figure_nrows = 10
figure_height_ratios = [0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.01, 0.4]
figure_width_ratios = [1., 1., 1., 1., 1., 1., 1., 1.]
dpi = 500

#
color0 = '#a6b96f'
color1 =  '#53589c'
color_both = '#d4b565'

marker0 = 's'
marker1 = 'o'
marker_both = 'd'

markersize0 = 4
markersize1 = 4
markersize_both = 4

alpha = 0.75

# fontsizes
labelfontsize = 10
tickfontsize = 8

# Binning Parameters
nbins = 10

# Legend
fontsize_legend = 9
ncol_legend = 3
pos_legend = (0.5, -0.15)

### MAIN ###

## Preprocessing
results = {}

for hg in hypergraphs:

    print(hg)
    results[hg] = {}

    # load the hypergraph
    graphtype = "pairwise" if pairwise else "hypergraph"
    with open(f'{results_dir}/{hg}_{graphtype}_degree.pkl', 'rb') as f:

        degree_sequence = pickle.load(f)

    n = np.max(list(degree_sequence.keys())) + 1
    degree_sequence = np.asarray([degree_sequence[i] for i in range(n)])

    # load the group assignments
    with open(f'{results_dir}/{hg}_hypergraph_group.pkl', 'rb') as f:

        group_assignments = pickle.load(f)
    
    if type(group_assignments)==np.ndarray:

        degree_sequence0 = []
        degree_sequence1 = []
        for j in range(group_assignments.shape[0]):
            degree_sequence0.append(degree_sequence[np.where(group_assignments[j, :]==0)])
            degree_sequence1.append(degree_sequence[np.where(group_assignments[j, :]==1)])

        degree_sequence0 = np.concatenate(degree_sequence0)
        degree_sequence1 = np.concatenate(degree_sequence1)

    else:
        group_assignments = np.asarray(group_assignments)
        degree_sequence0 = degree_sequence[np.where(group_assignments==0)]
        degree_sequence1 = degree_sequence[np.where(group_assignments==1)]

    # binning
    pk, k = degree_distribution_binning(degree_sequence, nbins=nbins, logbinning=hypergraph_settings[hg]['logbin'])
    pk0, k0 = degree_distribution_binning(degree_sequence0, nbins=nbins, logbinning=hypergraph_settings[hg]['logbin'])
    pk1, k1 = degree_distribution_binning(degree_sequence1, nbins=nbins, logbinning=hypergraph_settings[hg]['logbin'])

    # store
    results[hg]['k'] = k
    results[hg]['pk'] = pk

    results[hg]['k0'] = k0
    results[hg]['pk0'] = pk0

    results[hg]['k1'] = k1
    results[hg]['pk1'] = pk1

## Create the main figure
fig, gs = create_figure(
    figsize=figsize,
    nrows=figure_nrows,
    ncols=figure_ncols,
    height_ratios=figure_height_ratios,
    width_ratios=figure_width_ratios,
    hspace=figure_hspace,
    wspace=figure_wspace
)

## Create a header
fig = add_text(fig, gs[0,:], text_line, text_coords, text_style)


## Plot the different hypergraphs
for i, hg in enumerate(hypergraphs):

    # create degree distribution
    c = coordinates[i] 
    ax = fig.add_subplot(gs[c[0], c[1]:c[2]])

    ax.plot(
        results[hg]['k'],
        results[hg]['pk'],
        ls='',
        marker=marker_both,
        color=color_both,
        alpha=alpha
    )

    ax.plot(
        results[hg]['k0'],
        results[hg]['pk0'],
        ls='',
        marker=marker0,
        color=color0,
        alpha=alpha
    )

    ax.plot(
        results[hg]['k1'],
        results[hg]['pk1'],
        ls='',
        marker=marker1,
        color=color1,
        alpha=alpha
    )

    if hypergraph_settings[hg]['logbin']:
        ax.set_xscale('log')
        ax.set_yscale('log')



    if c[1] in [0, 1]:
        ax.set_ylabel(r"$p(k')$", fontsize=labelfontsize)

    else:
        ax.set_yticks([])

    ax.set_xticks(hypergraph_settings[hg]['xticks'])
    ax.tick_params(labelsize=tickfontsize)
    
    ax.set_xlim(hypergraph_settings[hg]['xlim'])
    ax.set_xlabel(r"$k'$", fontsize=labelfontsize)

    ax.set_ylim(hypergraph_settings[hg]['ylim'])
    ax.set_yticks(hypergraph_settings[hg]['yticks'])

    ax.minorticks_off()

    for xy, text, fs, fw, c in zip(annotation_coords, [abc[c], hypergraph_settings[hg]['name']], annotation_fontsizes, annotation_fontweights, annotation_colors):
        ax.text(*xy, text, ha='center', transform=ax.transAxes, fontsize=fs, fontweight=fw, color=c)

## create the legend
handles = [
    mpl.lines.Line2D([0], [0], marker=marker0, color=color0, linestyle='', markersize=markersize0, label=r'$p(k\mid g=0)$'),
    mpl.lines.Line2D([0], [0], marker=marker1, color=color1, linestyle='', markersize=markersize1, label=r'$p(k\mid g=1)$'),
    mpl.lines.Line2D([0], [0], marker=marker_both, color=color_both, linestyle='', markersize=markersize_both, label=r'$p(k)$'),
]

ax = fig.add_subplot(gs[-1, :])
ax.axis('off')
ax.legend(handles=handles, ncol=ncol_legend, loc='lower center', bbox_to_anchor=pos_legend, fontsize=fontsize_legend)

# save figure
fig.savefig(output_path +  figname + '.svg',  bbox_inches='tight') 
fig.savefig(output_path +  figname + '.pdf',  bbox_inches='tight')
fig.savefig(output_path +  figname + '.png', dpi=dpi,  bbox_inches='tight')