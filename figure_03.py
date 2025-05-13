"""
Figure 3

This script produces figure 03 in the main text,
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
output_path = './'
figname = 'fig03'
dpi = 500

text_style = {
    'fontsize'   : 10,
    'fontcolor'  : '#14191a',
    'fontweight' : 'bold'
}

text_coords = {
    0 : (-0.10, 0.20),
    2 : (-0.10, 0.15),
}

text_line_1 = "Acquisition fairness: Ability to receive information"
text_line_2 = "Diffusion fairness: Ability to disseminate information"

# Annotations
abc = {
    (1 ,0) : "a",
    (1 ,1) : "b",
    (1 ,2) : "c",
    (1 ,3) : "d",
    (3 ,0) : "e",
    (3 ,1) : "f",
    (3 ,2) : "g",
    (3 ,3) : "h"
}

annotation_fontsizes = [12, 10]
annotation_fontweights = ['bold', 'normal']
annotation_coords = {
    1  : [(-0.10, 1.1), (0.50, 1.1)],
    3  : [(-0.10, 1.1), (0.50, 1.1)],
}

annotation_colors = ['k', '#272c2d']
small_annotation_color = '#525758'

# Simulation Parameters
num_nodes = 10_000
spm = 0.25

homophily = ['neutral', 'weak', 'het']

# Bootstrap Parameters
bootstrap_seed = 213
bootstrap_num = 100
bootstrap_p = 0.99

# Figure Parameters
figsize = (7.2, 5.4)   # inches
figure_hspace = 0.75
figure_wspace = 0.25
figure_ncols = 4
figure_nrows = 5
figure_height_ratios = [0.5, 2.0, 0.5, 2.0, 0.25]
figure_width_ratios = [1., 1., 1., 1.]

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
        'weak' : '#234f77',
        'neutral' : '#5b6162',
        'het'  : '#e36900',
    },
    'colors_fill' : {
        'weak' : '#234f77',
        'neutral' : '#5b6162',
        'het'  : '#e36900',
    },
    'colors_axis' : 'k',
    'alpha_line' : 0.95,
    'linewidth' : 1,
    'alpha_fill' : 0.5,
    'linewidth_axis' : 1,
    'linestyle_axis' : '--',
    'fontsize_xlabels' : 10,
    'fontsize_ylabels' : 10,
    'fontsize_xticks'  : 8,
    'fontsize_yticks' : 8,
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
    'arrow_tip' :  (0.45, 0.85),
    'arrow_tail' : (0.45, 0.85),
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
    'arrow_tip' :  (0.45, 0.12),
    'arrow_tail' : (0.45, 0.12),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.0}
    }
]

# Diffusion Fairness Parameters

diffusion_xlabel = r'$f$'
diffusion_ylabel = r'$\delta(f)$'
diffusion_ylim = (0., 1.6)
diffusion_yticks = [0.0, 0.5, 1.0, 1.5]
diffusion_yticklabels = ['0.0', '0.5', '1.0', '1.5']
diffusion_xlim = (0., 0.9)
diffusion_xticks = [0.0, 0.3, 0.6, 0.9]
diffusion_xticklabels = ['0.0', '0.3', '0.6', '0.9']

diffusion_style = {
    'colors_line' : {
        'weak' : '#234f77',
        'neutral' : '#5b6162',
        'het'  : '#e36900',
    },
    'colors_fill' : {
        'weak' : '#234f77',
        'neutral' : '#5b6162',
        'het'  : '#e36900',
    },
    'colors_axis' : 'k',
    'alpha_line' : 0.95,
    'linewidth' : 1,
    'alpha_fill' : 0.5,
    'linewidth_axis' : 1,
    'linestyle_axis' : '--',
    'fontsize_xlabels' : 10,
    'fontsize_ylabels' : 10,
    'fontsize_xticks'  : 8,
    'fontsize_yticks' : 8,
}

diffusion_annotations = [
    {
    'text' : '',
    'text_fontsize': 6,
    'text_color' : small_annotation_color,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (0.9, 1.05),
    'arrow_tail' : (0.9, 0.75),
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
    'arrow_tip' :  (0.45, 0.80),
    'arrow_tail' : (0.45, 0.80),
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
    'arrow_tip' :  (0.9, 0.15),
    'arrow_tail' : (0.9, 0.45),
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
    'arrow_tip' :  (0.45, 0.22),
    'arrow_tail' : (0.45, 0.22),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.0}
    }
]

# Legend
legend_labels = [
        'homophilous',
        'neutral',
        'heterophilous'
]

legend_linecolors = [
        acquisition_style['colors_line']['weak'],
        acquisition_style['colors_line']['neutral'],
        acquisition_style['colors_line']['het'],
]

legend_facecolors = [
        acquisition_style['colors_fill']['weak'],
        acquisition_style['colors_fill']['neutral'],
        acquisition_style['colors_fill']['het'],
]

legend_fontsize = 7
legend_ncols = 3
legend_coords = (0.5, 0.05)
legend_columnspacing = 2.0


### MAIN ###

# initialize the random number generator for bootstrap
rng = np.random.default_rng(seed=bootstrap_seed)

## Load the data
with open(input_file, 'rb') as f:

    input_dict = pickle.load(f)

result = input_dict['result']
dynamics = input_dict['params']['dynamics']

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
    sub_data = {key[2] : val for key, val in result.items() if key[0]==spm and key[1]==dyn and key[2] in homophily}
    sub_data_0 = {key[2] : val for key, val in result.items() if key[0]==0 and key[1]==dyn and key[2] in homophily}
    sub_data_1 = {key[2] : val for key, val in result.items() if key[0]==1 and key[1]==dyn and key[2] in homophily}
 
    # add the acquisition fairness header
    if i==0:
        fig = add_text(fig, gs[0,:], text_line_1, text_coords[0], text_style)
        annotation_list = acquisition_annotations
    else:
        annotation_list = []

    # acquisition fairness plot for a given dynamics
    create_acquisition_fairness_plot(
        fig,
        gs[1,i],
        sub_data,
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
        [abc[(1,i)], dyn],
        annotation_coords[1],
        annotation_fontsizes,
        annotation_fontweights,
        annotation_colors,
        annotation_list,
        acquisition_style
    )

    # add the diffusion fairness header
    if i==0:
        fig = add_text(fig, gs[2,:], text_line_2, text_coords[0], text_style)
        annotation_list = diffusion_annotations
    else:
        annotation_list = []

    # diffusion fairness plot for a given dynamics
    create_diffusion_fairness_plot(
        fig,
        gs[3,i],
        sub_data_0,
        sub_data_1,
        diffusion_xlim,
        diffusion_ylim,
        bootstrap_p,
        bootstrap_num,
        rng,
        yaxis_visible,
        diffusion_xlabel,
        diffusion_ylabel,
        diffusion_yticks,
        diffusion_yticklabels,
        diffusion_xticks,
        diffusion_xticklabels,
        [abc[(3,i)], dyn],
        annotation_coords[3],
        annotation_fontsizes,
        annotation_fontweights,
        annotation_colors,
        annotation_list,
        diffusion_style
    )

    if i==0:
        fig = add_legend(
            fig,
            gs[4,:],
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






