"""
Figure 4 SI - 02

This script produces the analogoue of figure 04 in the
main text for the supplementary information. The input
needs to be prepared with prepare_data_synthetic.py.

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
stats_dir = './stats'
output_path = './'
figname = 'fig04_SI_02'

# Figure Parameters
figsize = (7.2, 6.2)   # inches
figure_hspace = 0.75
figure_wspace = 0.25
figure_ncols = 4
figure_nrows = 6
figure_height_ratios = [0.1, 1.0, 1.0, 1.0, 1.0, 0.05]
figure_width_ratios = [1., 1., 1., 1.]
dpi = 500

# Text Parameters
text_line = "Homophily patterns of real-world hypergraphs"
text_coords = (-0.15, 0.00)
text_style = {
    'fontsize' : 10,
    'fontcolor' : '#14191a',
    'fontweight' : 'bold'
}

# Annotation Parameters
letter_fs = 11
name_fs = 9
s_fs = 10

letter_fw = 'bold'
name_fw = 'normal'
s_fw = 'normal'

letter_coord = (-0.55, 0.75)
name_coord = (-0.55, 0.45)
s_coord = (0.5, 1.05)

letter_c = 'k'
name_c = '#272c2d'
s_c = 'k'

s_dict = {
    2 : r'$s = 2$',
    3 : r'$s = 3$',
    4 : r'$s = 4$',
    5 : r'$s = 5$'
}

abc = {
    (1,0) : "(a)",
    (1,1) : "",
    (1,2) : "",
    (1,3) : "",
    (2,0) : "(b)",
    (2,1) : "",
    (2,2) : "",
    (2,3) : "",
    (3,0) : "(c)",
    (3,1) : "",
    (3,2) : "",
    (3,3) : "",
    (4,0) : "(d)",
    (4,1) : "",
    (4,2) : "",
    (4,3) : "",
    (5,0) : "(e)",
    (5,1) : "",
    (5,2) : "",
    (5,3) : "",
    (6,0) : "(f)",
    (6,1) : "",
    (6,2) : "",
    (6,3) : "",
}

# Other Parameters
hypergraphs = [
    'aps_genderizerio',
    'dblp_genderizerio',
    'housebillsgender_genderizerio',
    'senatebillsgender_genderizerio'
]

hypergraph_names = [
    'APS',
    'DBLP',
    'House\n(Gender)',
    'Senate\n(Gender)',
]

groups = [0,1]
edge_sizes = [2,3,4,5]
q_low = 5
q_high = 95

# Barplot Parameters
bar_style = {
    'e_color' : 'k',
    'e_capsize' : 2,
    'e_linewidth' : 1.5,
    'axis_linestyle' : '--',
    'axis_linewidth' : 1,
    'axis_color' : '#363737',
    'fontsize_xticks' : 7,
    'fontsize_xlabel' : 10,
    'fontsize_ylabel' : 10,
    'fontsize_yticks' : 7,
    'bar_color' : ('#c2d58a','#6d71b7'),
    'bar_linecolor' : ('#a6b96f', '#53589c'),
    'bar_linewidth' : 0.75,
    'bar_width' : 0.25,
    'label_rotation' : 0,
    'ylabel_coords' : (-0.05, 0.15),
    'yticks_if_label' : [0.0, 1.0]
}

bar_xlabel = r'$r$'
bar_ylabel = r'$h^{(g)}_{s,r}$'

# Legend
legend_labels = [
        r'majority$\,g=0$',
        r'minority$\,g=1$',
]

legend_linecolors = [
        bar_style['bar_linecolor'][0],
        bar_style['bar_linecolor'][1]
]

legend_facecolors = [
        bar_style['bar_color'][0],
        bar_style['bar_color'][1]
]

legend_fontsize = 7
legend_ncols = 2
legend_coords = (0.82, 0.)
legend_columnspacing = 0.5


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

## Create a heaader
fig = add_text(fig, gs[0,:], text_line, text_coords, text_style)

for i, hg in enumerate(hypergraphs):

    with open(f"{stats_dir}/{hg}.pkl", "rb") as f:
        
        stats_dict = pickle.load(f)

    n_hg = stats_dict['num_nodes'].shape[0]
    h_mean = {g : {s : np.nanmean(stats_dict['homophily'][g][s], axis=0) for s in edge_sizes} for g in groups}
    h_perc = {g : {s : np.nanpercentile(stats_dict['homophily'][g][s], [q_low, q_high], axis=0) for s in edge_sizes} for g in groups}
    
    for j, s in enumerate(edge_sizes):

        # Decide on annotation
        if i==0 and j==0:
                annotation_texts = [abc[(i+1,j)], hypergraph_names[i], s_dict[s]]
                annotation_coords = [letter_coord, name_coord, s_coord]
                annotation_fontsizes = [letter_fs, name_fs, s_fs]
                annotation_fontweights = [letter_fw, name_fw, s_fw]
                annotation_colors = [letter_c, name_c, s_c]
        elif i==0 and j>0:
                annotation_texts = [abc[(i+1,j)], s_dict[s]]
                annotation_coords = [letter_coord, s_coord]
                annotation_fontsizes = [letter_fs, s_fs]
                annotation_fontweights = [letter_fw, s_fw]
                annotation_colors = [letter_c, s_c]
        elif i>0 and j==0:
                annotation_texts = [abc[(i+1,j)], hypergraph_names[i]]
                annotation_coords = [letter_coord, name_coord]
                annotation_fontsizes = [letter_fs, name_fs]
                annotation_fontweights = [letter_fw, name_fw]
                annotation_colors = [letter_c, name_c]
        else:
            annotation_texts = [abc[(i+1,j)]]
            annotation_coords = [letter_coord]
            annotation_fontsizes = [letter_fs]
            annotation_fontweights = [letter_fw]
            annotation_colors = [letter_c]

        if i==len(hypergraphs)-1:
            xlabel=bar_xlabel
        else:
            xlabel = None
        
        if j==0:
            yaxis_visible = True
            ylabel = bar_ylabel
        else:
            yaxis_visible = False
            ylabel = None
            
        # create the plot
        fig=create_homophily_barplot(
                fig,
                gs[i+1,j],
                h_mean,
                h_perc,
                s,
                n_hg,
                xlabel,
                ylabel,
                yaxis_visible,
                annotation_texts,
                annotation_coords,
                annotation_fontsizes,
                annotation_fontweights,
                annotation_colors,
                bar_style
                )

# put the legend
fig = add_legend(
    fig,
    gs[0,:],
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