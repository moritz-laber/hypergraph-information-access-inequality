"""
Figure 6

This script generates figure 06 in the main text,
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
figname = 'fig06'
dpi = 500

# Style Parameters
text_style = {
    'fontsize'   : 10,
    'fontcolor'  : '#14191a',
    'fontweight' : 'bold'
}

text_coords = {
    0 : (-0.05, 0.20),
    3 : (-0.05, 0.10),
    6 : (-0.05, 0.10),
    9 : (-0.05, 0.10)
}

text_lines = {
    0 : "Homophilous and neutral interactions",
    3 : "",
    6 : "Heterophilous and neutral interactions ",
    9 : ""
}

# Annotations
abc = {
    (2 ,0) : "(a)",
    (2 ,1) : "(b)",
    (2 ,2) : "(c)",
    (2 ,3) : "(d)",
    (5 ,0) : "(e)",
    (5 ,1) : "(f)",
    (5 ,2) : "(g)",
    (5 ,3) : "(h)",
    (8 ,0) : "(i)",
    (8 ,1) : "(j)",
    (8 ,2) : "(k)",
    (8 ,3) : "(l)",
    (11,0) : "(m)",
    (11,1) : "(n)",
    (11,2) : "(o)",
    (11,3) : "(p)",
}

annotation_fontsizes = [12, 10]
annotation_fontweights = ['bold', 'normal']
annotation_coords = {
    2  : [(-0.10, 1.0), (0.50, 1.0)],
    5  : [(-2.50, 1.0), (-0.20, 1.0)],
    8  : [(-0.10, 1.0), (0.50, 1.0)],
    11 : [(-2.50, 1.0), (-0.20, 1.0)],
}

annotation_colors = ['k', '#272c2d']

small_annotation_color = '#525758'

# Simulation Parameters
num_nodes = 10_000
spm = 0.25

homophily_pattern_list = [
    ['neutral', 'neutralweak3', 'weakneutral3', 'weak'],
    ['neutral', 'neutralhet3', 'hetneutral3', 'het']
]

order = [
    ['weak', 'neutralweak3', 'weakneutral3', 'neutral'],
    ['het' , 'neutralhet3', 'hetneutral3', 'neutral']
]

ridge_labels = {
    'weak': 'hom.',
    'neutral': 'neu.',
    'het': 'het.',
    'neutralweak3' : 'neu./hom.', 
    'weakneutral3' : 'hom./neu.',
    'neutralhet3' : 'neu./het.',
    'hetneutral3' : 'het./neu.',
    'hetweak3'    : 'het./hom.',
    'weakhet3'    : 'hom./het.'
}

violin_labels = {
    'weak': 'hom.',
    'neutral': 'neu.',
    'het': 'het.',
    'neutralweak3' : 'neu.\nhom.', 
    'weakneutral3' : 'hom.\nneu.',
    'neutralhet3' : 'neu.\nhet.',
    'hetneutral3' : 'het.\nneu.',
    'hetweak3'    : 'het.\nhom.',
    'weakhet3'    : 'hom.\nhet.'
}

# Figure Parameters
figsize = (7.2, 8.8)   # inches
figure_hspace = 0.75
figure_wspace = 0.25
figure_ncols = 4
figure_nrows = 13
figure_height_ratios = [0.01, 0.01, 1.75, 0.05, 0.1, 2.25, 0.2, 0.6, 1.75, 0.05, 0.1, 2.25, 1.0]
figure_width_ratios = [1., 1., 1., 1.]

# Ridge Plot Parameters
ridge_hspace = -0.5
ridge_wspace = 0.0
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
        'het'  : '#a63800',
        'neutralweak3' : '#789dcb',
        'weakneutral3' : '#a2c7f6',
        'neutralhet3'  : '#db6300',
        'hetneutral3'  : '#fe892c',
        'weakhet3'     : '#b0613b',
        'hetweak3'     : '#645664',
        'axis' : 'k'
    },
    'colors_face' : {
        'weak' : '#3d658e',
        'neutral' : '#6c7173',
        'het'  : '#ba4800',
        'neutralweak3' : '#8db2e0',
        'weakneutral3' : '#b9ddff',
        'neutralhet3'  : '#f07310',
        'hetneutral3'  : '#ffa448',
        'weakhet3'     : '#9f5f46',
        'hetweak3'     : '#79595b',
        'axis' : 'k'
    },
    'alpha_line' : 0.95,
    'alpha_face' : 0.95,
    'linewidth_axis' : 1,
    'fontsize_xlabels' : 9,
    'fontsize_xticks'  : 9,
    'fontsize_yticks' : 8,
}


ridge_annotations = [
    {
    'text' : '',
    'text_fontsize': 6,
    'text_color' : small_annotation_color,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (-0.20, -0.9),
    'arrow_tail' : (0.05, -0.9),
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
    'arrow_tip' :  (-0.18, -1.55),
    'arrow_tail' : (-0.18, -1.55),
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
    'arrow_tip' :  (1.15, -0.9),
    'arrow_tail' : (0.90, -0.9),
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
    'arrow_tip' :  (0.95, -1.55),
    'arrow_tail' : (0.95, -1.55),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.0}
    }
]
    
                

# Violin Plot Parameters
violin_hspace = 0.
violin_wspace = 0.
violin_ylabel = r'$t^{(g)}_{90}$'
violin_threshold = 1e-3

violin_xmin = 0.
violin_xmax = 125.
violin_ymin = 0.
violin_ymax = None
violin_npoints = 200

violin_style = {
    'colors_line' : { 
        'weak' : ('#002547', '#002547'),
        'neutral' : ('#5b6162', '#5b6162'),
        'het'  : ('#a63800', '#a63800'),
        'neutralweak3' : ('#789dcb', '#789dcb'),
        'weakneutral3' : ('#a2c7f6', '#a2c7f6'),
        'neutralhet3'  : ('#db6300', '#db6300'),
        'hetneutral3'  : ('#fe892c', '#fe892c'),
        'weakhet3'     : ('#b0613b', '#b0613b'),
        'hetweak3'     : ('#645664','#645664'),
        'axis' : 'k'
    },
    'colors_face' : { 
        'weak' : ('#043a5f', '#043a5f'),
        'neutral' : ('#6c7173', '#6c7173'),
        'het'  : ('#ba4800', '#ba4800'),
        'neutralweak3' : ('#789dcb', '#789dcb'),
        'weakneutral3' : ('#a2c7f6', '#a2c7f6'),
        'neutralhet3'  : ('#db6300', '#db6300'),
        'hetneutral3'  : ('#fe892c', '#fe892c'),
        'weakhet3'     : ('#b0613b', '#b0613b'),
        'hetweak3'     : ('#645664','#645664'),
        'axis' : 'k'
    },
    'alpha_line' : 0.95,
    'alpha_face' : 0.95,
    'linewidth_axis' : 1,
    'fontsize_xticks' : 7,
    'fontsize_yticks' : 8,
    'fontsize_ylabels' : 10,
    'label_rotation' : 40
}

violin_annotations = [
    {
    'text' : 'maj.',
    'text_color' : small_annotation_color,
    'text_fontsize': 6,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (-0.15, 0.58),
    'arrow_tail' : (-0.15, 0.58),
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
    'arrow_tip' :  (0.60, 0.58),
    'arrow_tail' : (0.60, 0.58),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.0}
    }
]


# Legend
legend_labels = [
        r'homophily$\,s\in\{2,3,4\}$',
        r'heterophily$\,s\in\{2,3,4\}$',
        r'homophily$\,s=2,\,$neutral$\,s\in\{3,4\}$',
        r'heterophily$\,s=2,\,$neutral$\,s\in\{3,4\}$',
        r'neutral$\,s=2,\,$homophily$\,s\in\{3,4\}$',
        r'neutral$\,s=2,\,$heterophily$\,s\in\{3,4\}$',
        r'neutral$\,s\in\{2,3,4\}$'
]

legend_linecolors = [
        ridge_style['colors_line']['weak'],
        ridge_style['colors_line']['het'],
        ridge_style['colors_line']['weakneutral3'],
        ridge_style['colors_line']['hetneutral3'],
        ridge_style['colors_line']['neutralweak3'],
        ridge_style['colors_line']['neutralhet3'],
        ridge_style['colors_line']['neutral'],
    
]

legend_facecolors = [
        ridge_style['colors_face']['weak'],
        ridge_style['colors_face']['het'],
        ridge_style['colors_face']['weakneutral3'],
        ridge_style['colors_face']['hetneutral3'],
        ridge_style['colors_face']['neutralweak3'],
        ridge_style['colors_face']['neutralhet3'],
        ridge_style['colors_face']['neutral'],
]

legend_fontsize = 7
legend_ncols = 4
legend_columnspacing = 0.5
legend_coords = (0.5, 0.05)

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

# iterate over columns of the figure
for i, dyn_nu_lam in enumerate(dynamics):
    
    # name of the dynamics 
    dyn = dyn_nu_lam[0]

    # keep the y-axis visible only in the first column
    if i==0:
        yaxis_visible=True
    else:
        yaxis_visible=False
    

    # iterate over the rows of the figure
    for j, homophily_patterns in enumerate(homophily_pattern_list):
        
        if j==0:
            row = 0
        elif j==1:
            row = 6
        else:
            print(j)
    
        # subset the data
        ridge_kdes_sub = {key[2] : val for key, val in ridge_kdes.items() if key[0]==spm and key[1]==dyn and key[2] in homophily_patterns}
        violin_kdes_sub = {key[2] : val for key, val in violin_kdes.items() if key[0]==spm and key[1]==dyn and key[2] in homophily_patterns}

        # create header
        if i==0:
            fig = add_text(fig, gs[row:row+2,:], text_lines[row], text_coords[row], text_style)
            annotation_list = ridge_annotations
        
        else:
            annotation_list = []
        
        # create ridgeplot
        fig = create_ridgeplot(
            fig,
            gs[row+2, i],
            ridge_vals,
            ridge_kdes_sub,
            ridge_xlimits,
            ridge_ylimits,
            ridge_wspace,
            ridge_hspace,
            order[j],
            yaxis_visible,
            ridge_xlabel,
            ridge_xticks,
            ridge_xticklabels,
            ridge_labels,
            [abc[(row+2,i)], dyn],
            annotation_coords[row+2],
            annotation_fontsizes,
            annotation_fontweights,
            annotation_colors,
            annotation_list,
            ridge_style
        )

        if i==0:
            fig = add_text(fig, gs[row+3:row+5,:], text_lines[row+3], text_coords[row], text_style)
            annotation_list = violin_annotations
        else:
            annotation_list = []
    
        # create violinplot
        fig = create_violinplot(
            fig,
            gs[row+5,i],
            violin_vals,
            violin_kdes_sub,
            violin_xlimits,
            violin_ylimits,
            violin_wspace,
            violin_hspace,
            yaxis_visible,
            order[j],
            violin_labels,
            violin_threshold,
            violin_ylabel,
            [abc[(row+5,i)], dyn],
            annotation_coords[row+5],
            annotation_fontsizes,
            annotation_fontweights,
            annotation_colors,
            annotation_list,
            violin_style
        )


        if i==0 and j==1:
            fig = add_legend(
                fig,
                gs[row+6,:],
                legend_facecolors,
                legend_linecolors,
                legend_labels,
                legend_ncols,
                legend_columnspacing,
                legend_coords,
                legend_fontsize
            )

# save figure
fig.savefig(output_path +  figname + '.svg',  bbox_inches='tight')
fig.savefig(output_path +  figname + '.pdf',  bbox_inches='tight')
fig.savefig(output_path +  figname + '.png', dpi=dpi,  bbox_inches='tight')