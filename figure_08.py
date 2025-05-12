"""
Figure 8

This script generates acquisition fairness plots
for real-world hypergraphs.

ML - 2025/05/12
"""

### IMPORT ###
import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import numpy as np
import pickle

from plot_functions import *

### PARAMETERS ###

# I/O Parameters
results_dir = './results'
stats_dir = './stats'
output_path = './'
figname = 'fig08'

hypergraphs = [
    'primaryschool',
    'highschool',
    'hospital',
    'housebills',
    'housebillsgender_genderapi',
    'aps_genderapi',
    'senatebills',
    'senatebillsgender_genderapi', 
    'dblp_genderapi'
]


hypergraph_names = [
    'Primary School',
    'High School',
    'Hospital',
    'House (Party)',
    'House (Gender)',
    'APS',
    'Senate (Party)',
    'Senate (Gender)',
    'DBLP'
]

order = [
    'linear',
    'sublinear',
    'superlinear',
    'asymmetric'
]

labels = {
    'linear'      : 'lin.',
    'sublinear'   : 'sub.',
    'superlinear' : 'sup.',
    'asymmetric'  : 'asym.'
}

coordinates = {
    0 : (1,0),
    1 : (1,1),
    2 : (1,2),
    3 : (3,0),
    4 : (3,1),
    5 : (3,2),
    6 : (5,0),
    7 : (5,1),
    8 : (5,2)
}

abc = {
    (1,0) : "a",
    (1,1) : "b",
    (1,2) : "c",
    (3,0) : "d",
    (3,1) : "e",
    (3,2) : "f",
    (5,0) : "g",
    (5,1) : "h",
    (5,2) : "i"
}

annotation_fontsizes = [12, 10]
annotation_fontweights = ['bold', 'normal']
annotation_coords = [(-0.1, 1.15), (0.45, 1.15)]

annotation_colors = ['k', '#272c2d']
small_annotation_color = '#525758'

spm  = 0

text_style = {
    'fontsize' : 10,
    'fontcolor' : '#14191a',
    'fontweight' : 'bold'
}

text_coords = (-0.04, 0.4)
text_line = "Acquisition fairness: Ability to acquire information"

# Figure Parameters
figsize = (7.2, 7.2)   # inches
figure_hspace = 0.75
figure_wspace = 0.45
figure_ncols = 3
figure_nrows = 8
figure_height_ratios = [0.2, 1.0, 0.15, 1.0, 0.15, 1.0, 0.01, 0.01]
figure_width_ratios = [1., 1., 1.]
dpi = 500

# Bootstrap Parameters
bootstrap_seed = 213
bootstrap_num = 100
bootstrap_p = 0.99

# Acquisition Fairness Parameters
acquisition_xlabel = r'$f$'
acquisition_ylabel = r'$\alpha(f)$'
acquisition_ylim = (0., 1.6)
acquisition_yticks = [0.0, 0.5, 1.0, 1.5]
acquisition_yticklabels = ['0.0', '0.5', '1.0', '1.5']
acquisition_xlim = (0.0, 0.9)
acquisition_xticks = [0.0, 0.3, 0.6, 0.9]
acquisition_xticklabels = ['0.0', '0.3', '0.6', '0.9']

acquisition_style = {
    'colors_line' : {
        'linear' : '#004563',
        'sublinear' : '#b679ae',
        'superlinear'  : '#8c2b2c',
        'asymmetric' : '#647f1a',
    },
    'colors_fill' : {
        'linear' : '#015e7e',
        'sublinear' : '#d294ca',
        'superlinear'  : '#a74341',
        'asymmetric' : '#7f9a36',
    },
    'colors_axis' : 'k',
    'alpha_line' : 0.95,
    'alpha_fill' : 0.95,
    'linewidth' : 1,
    'linewidth_axis' : 1,
    'linestyle_axis' : '--',
    'fontsize_xlabels' : 10,
    'fontsize_ylabels' : 10,
    'fontsize_xticks'  : 9,
    'fontsize_yticks' : 9,
    'label_rotation' : 40 
}

acquisition_annotations = [
    {
    'text' : '',
    'text_fontsize': 6,
    'text_color' : small_annotation_color,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (0.9, 1.10),
    'arrow_tail' : (0.9, 0.80),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.75}
    },
    {
    'text' : 'minority\nadvantage',
    'text_color' : small_annotation_color,
    'text_fontsize': 6,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (0.55, 0.85),
    'arrow_tail' : (0.55, 0.85),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.0}
    },
    {
    'text' : '',
    'text_color' : small_annotation_color,
    'text_fontsize': 6,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (0.9, 0.05),
    'arrow_tail' : (0.9, 0.35),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.75}
    },
    {
    'text' : 'majority\nadvantage',
    'text_color' : small_annotation_color,
    'text_fontsize': 6,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (0.55, 0.12),
    'arrow_tail' : (0.55, 0.12),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.0}
    }
]



# Legend
legend_labels = [
        'linear',
        'sublinear',
        'superlinear',
        'asymmetric'
]

legend_linecolors = [
        acquisition_style['colors_line']['linear'],
        acquisition_style['colors_line']['sublinear'],
        acquisition_style['colors_line']['superlinear'],
        acquisition_style['colors_line']['asymmetric'],
]

legend_facecolors = [
        acquisition_style['colors_line']['linear'],
        acquisition_style['colors_line']['sublinear'],
        acquisition_style['colors_line']['superlinear'],
        acquisition_style['colors_line']['asymmetric'],
]

legend_fontsize = 7
legend_ncols = 4
legend_columnspacing = 0.5
legend_coords = (0.5, 0.05)

### MAIN ###

# initialize the random number generator for bootstrap
rng = np.random.default_rng(seed=bootstrap_seed)

## load the data
results = {}
for hg in hypergraphs:

    with open(f'{results_dir}/{hg}_plotdata.pkl', 'rb') as f:

        hg_result = pickle.load(f)

    for key, val in hg_result.items():

        key = (key[0], key[1], hg)
        
        results[key] = val


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

## Create a heaader
fig = add_text(fig, gs[0,:], text_line, text_coords, text_style)


## Plot the different hypergraphs
for i, hg in enumerate(hypergraphs):
    
    if coordinates[i] == (1,0):
        annotation_list = acquisition_annotations
    else:
        annotation_list = []

    if coordinates[i][1] == 0:
        yaxis_visible = True
    else:
        yaxis_visible = False
            
    sub_results = {key[1] : val for key, val in results.items() if key[0]==spm and key[2]==hg}

    # acquisition fairness plot for a given dynamics
    create_acquisition_fairness_plot(
        fig,
        gs[coordinates[i]],
        sub_results,
        acquisition_xlim,
        acquisition_ylim,
        bootstrap_p,
        bootstrap_num,
        rng,
        yaxis_visible,
        acquisition_xlabel,
        acquisition_ylabel,
        acquisition_yticks,
        acquisition_yticklabels,
        acquisition_xticks,
        acquisition_xticklabels,
        [abc[coordinates[i]], hypergraph_names[i]],
        annotation_coords,
        annotation_fontsizes,
        annotation_fontweights,
        annotation_colors,
        annotation_list,
        acquisition_style
    )


## create the legend
fig = add_legend(
    fig,
    gs[7,:],
    legend_facecolors,
    legend_linecolors,
    legend_labels,
    legend_ncols,
    legend_columnspacing,
    legend_coords,
    legend_fontsize,
)

# save figure
fig.savefig(output_path +  figname + '.svg',  bbox_inches='tight')
fig.savefig(output_path +  figname + '.pdf',  bbox_inches='tight')
fig.savefig(output_path +  figname + '.png', dpi=dpi,  bbox_inches='tight')