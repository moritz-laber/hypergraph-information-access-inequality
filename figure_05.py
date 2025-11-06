"""
Figure 5

This script generates figure 05 in the main text,
as well as the analogous figures in the SI. The 
input needs to be prepared with prepare_data_synthetic.py.

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
figname = 'fig05'

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
    (1,0) : "(a)",
    (1,1) : "(b)",
    (1,2) : "(c)",
    (3,0) : "(d)",
    (3,1) : "(e)",
    (3,2) : "(f)",
    (5,0) : "(g)",
    (5,1) : "(h)",
    (5,2) : "(i)"
}

annotation_fontsizes = [12, 10]
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

text_coords = (-0.05, 0.05)
text_line = "Information access inequality in real-world hypergraphs"

# Figure Parameters
figsize = (7.2, 7.2)   # inches
figure_hspace = 0.75
figure_wspace = 0.25
figure_ncols = 3
figure_nrows = 8
figure_height_ratios = [0.1, 1.0, 0.15, 1.0, 0.15, 1.0, 0.05, 0.05]
figure_width_ratios = [1., 1., 1.]
dpi = 500

# Ridge Plot Parameters
ridge_hspace = -0.5
ridge_wspace = 0.0
ridge_order = order
ridge_labels = labels
ridge_xlabel = r'$d_W(\mathcal{Z}^{(0)},\mathcal{Z}^{(1)})$'

ridge_xmin = 0.
ridge_ymin = 0.
ridge_ymax = None
ridge_npoints = 200

ridge_style = {
    'colors_line' : {
        'linear' : '#004563',
        'sublinear' : '#b679ae',
        'superlinear'  : '#8c2b2c',
        'asymmetric' : '#647f1a',
        'axis' : 'k'
    },
    'colors_face' : {
        'linear' : '#015e7e',
        'sublinear' : '#d294ca',
        'superlinear'  : '#a74341',
        'asymmetric' : '#7f9a36',
        'axis' : 'k'
    },
    'alpha_line' : 0.95,
    'alpha_face' : 0.95,
    'linewidth_axis' : 1,
    'fontsize_xlabels' : 10,
    'fontsize_xticks'  : 9,
    'fontsize_yticks' : 9,
}


ridge_annotations = [
    {
    'text' : '',
    'text_fontsize': 6,
    'text_color' : small_annotation_color,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (-0.07, -0.8),
    'arrow_tail' : (0.18, -0.8),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.75}
    },
    {
    'text' : 'more\nequal',
    'text_color' : small_annotation_color,
    'text_fontsize': 6,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (-0.02, -1.35),
    'arrow_tail' : (-0.02, -1.35),
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
    'arrow_tip' :  (1.07, -0.8),
    'arrow_tail' : (0.82, -0.8),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.75}
    },
    {
    'text' : 'less\nequal',
    'text_color' : small_annotation_color,
    'text_fontsize': 6,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (0.87, -1.35),
    'arrow_tail' : (0.87, -1.35),
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
        ridge_style['colors_line']['linear'],
        ridge_style['colors_line']['sublinear'],
        ridge_style['colors_line']['superlinear'],
        ridge_style['colors_line']['asymmetric']
]

legend_facecolors = [
        ridge_style['colors_face']['linear'],
        ridge_style['colors_face']['sublinear'],
        ridge_style['colors_face']['superlinear'],
        ridge_style['colors_face']['asymmetric']
]

legend_fontsize = 7
legend_ncols = 4
legend_coords = (0.5, 0.05)
legend_columnspacing = 2.0


### MAIN ###

## load the data
results = {}
num_nodes = {}

for hg in hypergraphs:

    with open(f'{results_dir}/{hg}_plotdata.pkl', 'rb') as f:

        hg_result = pickle.load(f)

    for key, val in hg_result.items():

        key = (key[0], key[1], hg)
        
        results[key] = val

    with open(f'{stats_dir}/{hg}.pkl', 'rb') as f:
        
        hg_stats = pickle.load(f)
    
    num_nodes[hg] = hg_stats['num_nodes'][0,2].astype('int')


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
    
    if coordinates[i][1] == 0:
        yaxis_visible = True
        if coordinates[i][0] == 1:
            annotation_list = ridge_annotations
        else:
            annotation_list = []
    else:
        yaxis_visible = False
        annotation_list = []

    sub_result = {key[1] : val for key, val in results.items() if key[0]==spm and key[2]==hg}

    # set x-axis limits for ridge plot
    if 'aps' in hg or 'dblp' in hg:
        ridge_xticks = [0.0, num_nodes[hg]/8., num_nodes[hg]/4.]
        ridge_xticklabels = [r'$0.0$', r'$n/8$', r'$n/4$']
        ridge_xmin = 0.
        ridge_xmax = num_nodes[hg]/4.
    else:
        ridge_xticks = [0.0, num_nodes[hg]/4., num_nodes[hg]/2.]
        ridge_xticklabels = [r'$0.0$', r'$n/4$', r'$n/2$']
        ridge_xmin = 0.
        ridge_xmax = num_nodes[hg]/2.

    ridge_kdes, ridge_vals, ridge_xlimits, r_ylimits = compute_kde_ridge(
            sub_result,
            npoints=ridge_npoints,
            xmin=ridge_xmin,
            xmax=ridge_xmax
    )

    # update axis limits for ridge plot
    ridge_ylimits = [0.,0.]
    if ridge_ymin:
        ridge_ylimits[0] = ridge_ymin
    else:
        ridge_ylimits[0] = r_ylimits[0]
    if ridge_ymax:
        ridge_ylimits[1] = ridge_ymax
    else:
        ridge_ylimits[1] = r_ylimits[1]
    ridge_ylimits = tuple(ridge_ylimits)

    # create ridgeplot
    fig = create_ridgeplot(
        fig,
        gs[coordinates[i]],
        ridge_vals,
        ridge_kdes,
        ridge_xlimits,
        ridge_ylimits,
        ridge_wspace,
        ridge_hspace,
        ridge_order,
        yaxis_visible,
        ridge_xlabel,
        ridge_xticks,
        ridge_xticklabels,
        ridge_labels,
        [abc[coordinates[i]], hypergraph_names[i]],
        annotation_coords,
        annotation_fontsizes,
        annotation_fontweights,
        annotation_colors,
        annotation_list,
        ridge_style
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