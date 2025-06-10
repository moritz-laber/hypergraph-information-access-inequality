"""
Figure 2

This script produces figure 02 in the main text,
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
input_file = './results.pkl'
output_path = '../'
figname = 'fig02'
dpi = 500


text_style = {
    'fontsize' : 10,
    'fontcolor' : '#14191a',
    'fontweight' : 'bold'
}

text_coords = [(-0.05, 0.15), (-0.05, 0.15), (-0.05, 0.25)]
text_line_1 = "Quantifying information access inequality with optimal transport"
text_line_2 = "Distribution of time to inform a given group"
text_line_3 = "Effect of higher-order homophily on information transmission paths"

verbose = True

# Annotations
abc = {
    (1,0) : "a",
    (1,1) : "b",
    (1,2) : "c",
    (1,3) : "d",
    (4,0) : "e",
    (4,1) : "f",
    (4,2) : "g",
    (4,3) : "h",
    (6,0) : "i",
    (6,1) : "j",
    (6,2) : "k",
    (6,3) : "l"
}

annotation_fontsizes = [12, 10]
annotation_fontweights = ['bold', 'normal']
annotation_coords = {
    1 : [(-0.10, 1.0), (0.50, 1.0)],
    4 : [(-1.35, 1.0), (0.50, 1.0)],
    6 : [(-0.10, 1.2), (0.50, 1.2)],
}

annotation_colors = ['k', '#272c2d']
small_annotation_color = '#525758'

# Simulation Parameters
num_nodes = 10_000
spm = 0.25
order = ['weak', 'neutral', 'het']
labels =  {'weak' : 'hom.', 'neutral' : 'neu.', 'het' : 'het.'}

# Figure Parameters
figsize = (7.2, 7.2)   # inches
figure_hspace = 0.75
figure_wspace = 0.25
figure_ncols = 4
figure_nrows = 9
figure_height_ratios = [0.05, 2.0, 0.5, 0.4, 2.5, 0.35, 0.35, 2, 1.5]
figure_width_ratios = [1., 1., 1., 1.]

# Ridge Plot Parameters
ridge_hspace = -0.5
ridge_wspace = 0.0
ridge_order = order
ridge_labels = labels
ridge_xlabel = r'$d_W(\mathcal{Z}^{(0)},\mathcal{Z}^{(1)})$'
ridge_xticks = [0., num_nodes/4., num_nodes/2.]
ridge_xticklabels = [r'$0$', r'$n/4$', r'$n/2$']

ridge_xmin = 0.
ridge_xmax = num_nodes/2.
ridge_ymin = 0.
ridge_ymax = None
ridge_npoints = 200

ridge_style = {
    'colors_line' : {
        'weak' : '#234f77',
        'neutral' : '#5b6162',
        'het'  : '#e36900',
        'axis' : 'k'
    },
    'colors_face' : {
        'weak' : '#3d658e',
        'neutral' : '#6c7173',
        'het'  : '#f87a06',
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
    'arrow_tip' :  (-0.20, -0.8),
    'arrow_tail' : (0.05, -0.8),
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
    'arrow_tip' :  (-0.18, -1.45),
    'arrow_tail' : (-0.18, -1.45),
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
    'arrow_tip' :  (1.15, -0.8),
    'arrow_tail' : (0.90, -0.8),
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
    'arrow_tip' :  (0.95, -1.45),
    'arrow_tail' : (0.95, -1.45),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.0}
    }
]
    
                

# Violin Plot Parameters
violin_hspace = 0.
violin_wspace = 0.
violin_order = order
violin_labels = labels
violin_ylabel = r'$t^{(g)}_{90}$'
violin_threshold = 1e-3

violin_xmin = 0.
violin_xmax = 125.
violin_ymin = 0.
violin_ymax = None
violin_npoints = 200

violin_style = {
    'colors_line' : { 
        'weak' : ('#002547', '#6e93c0'),
        'neutral' : ('#3c4142', '#a0a6a7'),
        'het'  : ('#a63800', '#ff9429'),
        'axis' : 'k'
    },
    'colors_face' : { 
        'weak' : ('#043a5f', '#87acd9'),
        'neutral' : ('#4b5152', '#b2b8b9'),
        'het'  : ('#ba4800', '#ffaf46'),
        'axis' : 'k'
    },
    'alpha_line' : 0.95,
    'alpha_face' : 0.95,
    'linewidth_axis' : 1,
    'fontsize_xticks' : 9,
    'fontsize_yticks' : 9,
    'fontsize_ylabels' : 10,
    'label_rotation' : 60
}

violin_annotations = [
    {
    'text' : 'maj.',
    'text_color' : small_annotation_color,
    'text_fontsize': 6,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (0.02, 0.58),
    'arrow_tail' : (0.02, 0.58),
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
    'arrow_tip' :  (0.58, 0.58),
    'arrow_tail' : (0.58, 0.58),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.0}
    }
]

# Barplot Parameters
bar_ylimits = (0.0, 0.51)

bar_width = 0.25
bar_yticks = [0., 0.25, 0.5]
bar_yticklabels = [r'$0$', r'$0.25$', r'$0.50$'] 
bar_xlabel=r'$s$'
bar_ylabel=r'$\eta_{s,g}$'
bar_normalize = num_nodes
bar_order = order
bar_style = {
    'colors_line' : { 
        'weak' : ('#002547', '#6e93c0'),
        'neutral' : ('#3c4142', '#a0a6a7'),
        'het'  : ('#a63800', '#ff9429'),
        'axis' : 'k'
    },
    'colors_face' : { 
        'weak' : ('#043a5f', '#87acd9'),
        'neutral' : ('#4b5152', '#b2b8b9'),
        'het'  : ('#ba4800', '#ffaf46'),
        'axis' : 'k'
    },
    'linewidth' : 0.5,
    'alpha_line' : 0.95,
    'alpha_face' : 0.95,
    'linewidth_axis' : 1,
    'fontsize_xlabels' : 10,
    'fontsize_ylabels' : 10,
    'fontsize_xticks' : 9,
    'fontsize_yticks' : 9
}

bar_annotations = [
    {
    'text' : 'maj.',
    'text_color' : small_annotation_color,
    'text_fontsize': 6,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (1.0, 0.23),
    'arrow_tail' : (1.0, 0.23),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'k',
        'lw':0.0}
    },
    {
    'text' : 'min.',
    'text_color' : small_annotation_color,
    'text_fontsize': 6,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (1.0, 0.58),
    'arrow_tail' : (1.0, 0.58),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'k',
        'lw':0.0}
    }
]

# Legend

legend_labels = [
        'minority, homophilous',
        'both, homophilous',
        'majority, homophilous',
        'minority, neutral',
        'both, neutral',
        'majority, neutral',
        'minority, heterophilous',
        'both, heterophilous',
        'majority, heterophilous'
]

legend_linecolors = [
        bar_style['colors_line']['weak'][1],
        ridge_style['colors_line']['weak'],
        bar_style['colors_line']['weak'][0],
        bar_style['colors_line']['neutral'][1],
        ridge_style['colors_line']['neutral'],
        bar_style['colors_line']['neutral'][0],
        bar_style['colors_line']['het'][1],
        ridge_style['colors_line']['het'],
        bar_style['colors_line']['het'][0]
]

legend_facecolors = [
        bar_style['colors_face']['weak'][1],
        ridge_style['colors_face']['weak'],
        bar_style['colors_face']['weak'][0],
        bar_style['colors_face']['neutral'][1],
        ridge_style['colors_face']['neutral'],
        bar_style['colors_face']['neutral'][0],
        bar_style['colors_face']['het'][1],
        ridge_style['colors_face']['het'],
        bar_style['colors_face']['het'][0]
]

legend_fontsize = 7
legend_ncols = 3
legend_coords = (0.5, 0.05)
legend_columnspacing = 2.0

### MAIN ###

## Load the data
with open(input_file, 'rb') as f:

    input_dict = pickle.load(f)

result = input_dict['result']
dynamics = input_dict['params']['dynamics']
homophily = input_dict['params']['homophily']

## fit kdes for Ridge
ridge_kdes, ridge_vals, ridge_xlimits, r_ylimits = compute_kde_ridge(
    result,
    npoints=ridge_npoints,
    xmin=ridge_xmin,
    xmax=ridge_xmax,
    verbose=verbose
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


## fit kdes for Violin
violin_kdes, violin_vals, violin_xlimits, v_ylimits = compute_kde_violin(
    result,
    npoints=violin_npoints,
    xmin=violin_xmin,
    xmax=violin_xmax
)

# update y limits for violin plot
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

## Fill the main figure
for i, dyn_nu_lam in enumerate(dynamics):

    # name of the dynamics 
    dyn = dyn_nu_lam[0]

    # keep the y-axis visible only in the first column
    if i==0:
        yaxis_visible=True
    else:
        yaxis_visible=False
    
    # subset the data
    bar_data = {key[2] : val for key, val in result.items() if key[0]==spm and key[1]==dyn}
    ridge_kdes_sub = {key[2] : val for key, val in ridge_kdes.items() if key[0]==spm and key[1]==dyn}
    violin_kdes_sub = {key[2] : val for key, val in violin_kdes.items() if key[0]==spm and key[1]==dyn}

    if i==0:
        fig = add_text(fig, gs[0,:], text_line_1, text_coords[0], text_style)
        annotation_list = ridge_annotations
    else:
        annotation_list = []

    # create ridgeplot
    fig = create_ridgeplot(
        fig,
        gs[1, i],
        ridge_vals,
        ridge_kdes_sub,
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
        [abc[(1,i)], dyn],
        annotation_coords[1],
        annotation_fontsizes,
        annotation_fontweights,
        annotation_colors,
        annotation_list,
        ridge_style
    )

    if i==0:
        fig = add_text(fig, gs[2:4,:], text_line_2, text_coords[1], text_style)
        annotation_list = violin_annotations
    else:
        annotation_list = []

    # create violinplot
    fig = create_violinplot(
        fig,
        gs[4,i],
        violin_vals,
        violin_kdes_sub,
        violin_xlimits,
        violin_ylimits,
        violin_wspace,
        violin_hspace,
        yaxis_visible,
        violin_order,
        violin_labels,
        violin_threshold,
        violin_ylabel,
        [abc[(4,i)], dyn],
        annotation_coords[4],
        annotation_fontsizes,
        annotation_fontweights,
        annotation_colors,
        annotation_list,
        violin_style
    )

    if i==0:
        fig = add_text(fig, gs[5:7,:], text_line_3, text_coords[2], text_style)
        annotation_list = bar_annotations
    else:
        annotation_list = []

    # create barplot
    fig = create_barplot(
        fig,
        gs[7, i],
        bar_data,
        bar_ylimits,
        bar_normalize,
        bar_width,
        bar_order,
        yaxis_visible,
        bar_xlabel,
        bar_ylabel,
        bar_yticks,
        bar_yticklabels,
        [abc[(6,i)], dyn],
        annotation_coords[6],
        annotation_fontsizes,
        annotation_fontweights,
        annotation_colors,
        annotation_list,
        bar_style
    )

    if i==0:
        fig = add_legend(
            fig,
            gs[8,:],
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