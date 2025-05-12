"""
Figure 7

This script generates the figure on informing
90% of the nodes in real-world hypergraphs.

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
figname = 'fig07'

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

percentile = [
    99,
    99,
    99,
    99,
    85,
    99,
    99,
    85,
    99
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
annotation_coords = [(-2.5, 1.15), (0., 1.15)]

annotation_colors = ['k', '#272c2d']
small_annotation_color = '#525758'

spm = 0.

text_style = {
    'fontsize' : 10,
    'fontcolor' : '#14191a',
    'fontweight' : 'bold'
}

text_coords = (-0.04, 0.10)
text_line = "Information access inequality in real-world hypergraphs"

# Figure Parameters
figsize = (7.2, 7.2)   # inches
figure_hspace = 0.75
figure_wspace = 0.45
figure_ncols = 3
figure_nrows = 8
figure_height_ratios = [0.2, 1.0, 0.15, 1.0, 0.15, 1.0, 0.05, 0.05]
figure_width_ratios = [1., 1., 1.]
dpi = 500

# Violin Plot Parameters
violin_hspace = 0.
violin_wspace = 0.
violin_order = order
violin_labels = labels
violin_ylabel = r'$t^{(g)}_{90}$'

violin_xmin = 0.
violin_xmax = None
violin_ymin = 0.
violin_ymax = None
violin_npoints = 400

violin_threshold = 0
yaxis_visible = True

violin_style = {
    'colors_line' : {
        'linear' : ('#004563', '#004563'),
        'sublinear' : ('#b679ae', '#b679ae'),
        'superlinear'  : ('#8c2b2c', '#8c2b2c'),
        'asymmetric' : ('#647f1a', '#647f1a'),
        'axis' : 'k'
    },
    'colors_face' : {
        'linear' : ('#015e7e', '#015e7e'),
        'sublinear' : ('#d294ca', '#d294ca'),
        'superlinear'  : ('#a74341', '#a74341'),
        'asymmetric' : ('#7f9a36', '#7f9a36'),
        'axis' : 'k'
    },
    'alpha_line' : 0.95,
    'alpha_face' : 0.95,
    'linewidth_axis' : 1,
    'fontsize_xlabels' : 10,
    'fontsize_ylabels' : 10,
    'fontsize_xticks'  : 9,
    'fontsize_yticks' : 9,
    'label_rotation' : 40 
}

violin_annotations = [
    {
    'text' : 'maj.',
    'text_color' : small_annotation_color,
    'text_fontsize': 6,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (-0.04, 0.88),
    'arrow_tail' : (-0.04, 0.88),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.0}
    },
    {
    'text' : 'min.',
    'text_color' : small_annotation_color,
    'text_fontsize': 6,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (0.6, 0.88),
    'arrow_tail' : (0.6, 0.88),
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
        violin_style['colors_line']['linear'][0],
        violin_style['colors_line']['sublinear'][0],
        violin_style['colors_line']['superlinear'][0],
        violin_style['colors_line']['asymmetric'][0]
]

legend_facecolors = [
        violin_style['colors_face']['linear'][0],
        violin_style['colors_face']['sublinear'][0],
        violin_style['colors_face']['superlinear'][0],
        violin_style['colors_face']['asymmetric'][0]
]

legend_fontsize = 7
legend_ncols = 4
legend_columnspacing = 0.5
legend_coords = (0.5, 0.05)

### MAIN ###

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
        annotation_list = violin_annotations
    else:
        annotation_list = []
            
    sub_result = {key[1] : val for key, val in results.items() if key[0]==spm and key[2]==hg}

    violin_kdes, violin_vals, violin_xlimits, v_ylimits = compute_kde_violin(
            sub_result,
            npoints=violin_npoints,
            xmin=violin_xmin,
            xmax=violin_xmax,
            percentile=percentile[i]
    )

    # update axis limits for ridge plot
    violin_ylimits = [0.,0.]
    if violin_ymin:
        violin_ylimits[0] = violin_ymin
    else:
        violin_ylimits[0] = v_ylimits[0]
    if violin_ymax:
        violin_ylimits[1] = violin_ymax
    else:
        violin_ylimits[1] = v_ylimits[1]
    violin_ylimits = tuple(violin_ylimits)

    # create ridgeplot
    fig = create_violinplot(
        fig,
        gs[coordinates[i]],
        violin_vals, 
        violin_kdes,
        violin_xlimits,
        violin_ylimits,
        violin_hspace,
        violin_wspace,
        yaxis_visible,
        order,
        violin_labels,
        violin_threshold,
        violin_ylabel,
        [abc[coordinates[i]], hypergraph_names[i]],
        annotation_coords,
        annotation_fontsizes,
        annotation_fontweights,
        annotation_colors,
        annotation_list,
        violin_style
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