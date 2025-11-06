"""
Figure: Gender-Labeling API

This script generates plots for comparing
the two gender labeling APIs on the APS 
and DBLP hypergraphs.

ML - 2025/05/12
"""

### IMPORT ###
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_functions import *

### PARAMETERS ###

# I/O Parameters
base_path = './metadata/'
output_path = './'
figname = 'fig_genderlabeling'

hypergraphs = [
    'aps',
    'dblp',
    'housebillsgender',
    'senatebillsgender'
]

names = [
    'APS',
    'DBLP',
    'House',
    'Senate'
]

abc = [
    "(a)",
    "(b)",
    "(c)",
    "(d)"
]

coordinates = [
    (1,0),
    (1,1),
    (3,0),
    (3,1)
]

header = 'Comparison of gender labeling APIs'

header_coords = (-0.03, 0.5)
header_style = {
    'fontsize'   : 10,
    'fontcolor'  : '#14191a',
    'fontweight' : 'bold'
}


# Annotations
text_fontsizes = [12, 10]
text_fontweights = ['bold', 'normal']
text_coords = [(-0.05, 1.15), (0.5, 1.15)]
text_colors = ['k', '#272c2d']

# Figure Parameters
figsize = (7.2, 5.2)   # inches
figure_hspace = 0.75
figure_wspace = 0.45
figure_ncols = 2
figure_nrows = 6
figure_height_ratios = [0.15, 1.0, 0.02, 1.0, 0.02, 0.05]
figure_width_ratios = [1., 1.,]
dpi = 500

xlabel = r'$p_v(g=1)$'
ylabel = r'$P(p_v(g=1))$'

ylabel_fontsize = 10
xlabel_fontsize = 10
ticklabelsize = 8 

e_linewidth = 1

bar_colors = [
    '#287c3720',
    '#d5ab0920' 
]

edge_colors = [
    '#287c37BF',
    '#d5ab09BF'
]



# Small Annotations
small_annotation_color = '#525758' 
small_annotations = [
    {
    'text' : '',
    'text_fontsize': 6,
    'text_color' : small_annotation_color,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (0.05, -0.25),
    'arrow_tail' : (0.25, -0.25),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.75}
    },
    {
    'text' : 'likely\nmale',
    'text_color' : small_annotation_color,
    'text_fontsize': 6,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (0.125, -0.45),
    'arrow_tail' : (0.125, -0.45),
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
    'arrow_tip' :  (0.95, -0.25),
    'arrow_tail' : (0.75, -0.25),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.75}
    },
    {
    'text' : 'likely\nfemale',
    'text_color' : small_annotation_color,
    'text_fontsize': 6,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (0.8, -0.45),
    'arrow_tail' : (0.8, -0.45),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.0}
    }
]

# Binning Parameters
pmin = 0.
pmax = 1.
nbins = 25
density = True

# Legend Parameters
ncols = 2
columnspacing = 1.5
legend_coords = (0.5, 0.15)
legend_fontsize = 8
legend_labels = [r'$\mathtt{GenderAPI}$',r'$\mathtt{genderize.io}$']

### MAIN ###

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

fig = add_text(fig, gs[0,:], header, header_coords, header_style)

## creating bins ##
bin_edges = np.linspace(pmin, pmax, num=int(nbins))
bin_mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])

## Loading the data
for i, hg in enumerate(hypergraphs):

    # Distribution Binning
    metadata_df = pd.read_csv(f'{base_path}metadata_{hg}.csv')

    p1_genderapi, _ = np.histogram(metadata_df['genderapi_p_female'].values, bins=bin_edges, density=density)
    p1_genderizerio, _ = np.histogram(metadata_df['genderizeio_p_female'].values, bins=bin_edges, density=density)
    
    ax = fig.add_subplot(gs[coordinates[i]])
    
    ax.bar(bin_mids, p1_genderapi, width=(pmax-pmin)/nbins, facecolor=bar_colors[0], edgecolor=edge_colors[0], linewidth=e_linewidth)
    ax.bar(bin_mids, p1_genderizerio, width=(pmax-pmin)/nbins, facecolor=bar_colors[1], edgecolor=edge_colors[1], linewidth=e_linewidth)

    # x-axis styling
    ax.set_xlim(pmin, pmax)
    if coordinates[i][0]==3:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    else:
        ax.set_xticks([])

    # y-axis styling
    if coordinates[i][1]==0:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    else:
        ax.set_yticks([])
    
    ax.tick_params(axis='both', labelsize=ticklabelsize)

    # remove spines and background
    spines = ["top", "right"]
    for s in spines:
        ax.spines[s].set_visible(False)

    ax.patch.set_alpha(0.)

    if coordinates[i]==(3,0):
        for annotation_dict in small_annotations:
            
            ax.annotate(
                annotation_dict['text'],
                color=annotation_dict['text_color'],
                fontsize=annotation_dict['text_fontsize'],
                fontstyle=annotation_dict['text_fontstyle'],
                xycoords='axes fraction',
                xy=annotation_dict['arrow_tip'],
                xytext=annotation_dict['arrow_tail'],
                arrowprops=annotation_dict['arrow_props']
            )


    # add annotations
    for text, coords, fs, fw, c in zip([abc[i], names[i]], text_coords, text_fontsizes, text_fontweights, text_colors):
        x,y = coords
        ax.text(x,y, text, ha='center', transform=ax.transAxes, fontsize=fs, fontweight=fw, color=c)
    
    
# add legend
fig = add_legend(
    fig,
    gs[5, :],
    bar_colors,
    edge_colors,
    legend_labels,
    ncols,
    columnspacing,
    legend_coords,
    legend_fontsize
)


# save figure
fig.savefig(output_path +  figname + '.svg',  bbox_inches='tight')
fig.savefig(output_path +  figname + '.pdf',  bbox_inches='tight')
fig.savefig(output_path +  figname + '.png', dpi=dpi,  bbox_inches='tight')
    