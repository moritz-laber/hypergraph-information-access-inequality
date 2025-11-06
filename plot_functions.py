"""
Plot Functions

This file contains the functions used to create
the plots with the scripts figure_[X].py.

ML, SD - 2025/05/12
"""

### IMPORT ###
import matplotlib as mpl
import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import os
import pandas as pd
import re
from scipy.stats import wasserstein_distance, gaussian_kde
from tqdm import tqdm
from typing import Tuple, Set, List, Optional, Dict, Callable

from simulation import get_params

### CONSTANTS ###
EDGE_SIZES = [2, 3, 4]     # the edge sizes that occur
GROUPS = [0, 1]            # the group labels
TIME_COLUMN = 1            # the index of the column that stores time at which information is received
GROUP_COLUMN = 2           # the index of the column that stores the group membership of the informed node
RANK_COLUMN = -1           # the index of the column that stores the rank of the informed node
EDGE_SIZE_COLUMN = 4       # the index of the column that stores the size of the edge responsible for transmission

### FUNCTIONS ###

## NEW FIGURE
def create_figure(
    figsize:Tuple[float],
    nrows:int, ncols:int,
    height_ratios:Optional[List[float]]=None,
    width_ratios:Optional[List[float]]=None,
    hspace=None,
    wspace=None
)->Tuple[mpl.figure.Figure, mpl.gridspec.GridSpec]:
    
    """Create new figure with gridspec.

    Input
    figsize - tuple describing the figure size in inches
    nrows   - number of rows in the gridspec
    ncols   - number of columns in the gridspec
    height_ratios - list of relative heights of gridspec panels
    width_ratios  - list of relative widths of gridspec panels
    hspace - gridspaceing horizontally
    wspace - gridspaceing vertically

    Output
    fig - figure
    gs  - gridspec
    """

    # create the figure
    fig = plt.figure(figsize=figsize)

    # create the gridspec
    gs = grid_spec.GridSpec(
        figure=fig,
        nrows=nrows,
        ncols=ncols,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=hspace,
        wspace=wspace
    )

    return fig, gs

## TITLE & EMPTY GRID CELLS
def add_text(
    fig:mpl.figure.Figure,
    gs:mpl.gridspec.GridSpec,
    text:str,
    pos:Tuple[float],
    style:Dict[str,...]
)->mpl.figure.Figure:
    
    """Add a new axis in the gridspec with text.

    Input
    fig - the figure
    gs  - the gridspec panel
    title - the title

    Output
    fig - figure
    """

    # add new axis
    ax = fig.add_subplot(gs)

    # turn of ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # turn of spines
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)

    # remove background
    ax.patch.set_alpha(0.)

    # set the title
    x,y = pos
    ax.text(x, y, text, transform=ax.transAxes, fontsize=style['fontsize'], fontweight=style['fontweight'], color=style['fontcolor'])

    return fig

# LEGEND 
def add_legend(
    fig:mpl.figure.Figure,
    gs:mpl.gridspec.GridSpec,
    facecolors:List[str],
    edgecolors:List[str],
    labels:List[str],
    ncols:int,
    columnspacing:float,
    coords:Tuple[float],
    fontsize:int
)->mpl.figure.Figure:

    # add new axis
    ax = fig.add_subplot(gs)

    # turn of ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # turn of spines
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)

    # remove background
    ax.patch.set_alpha(0.)

    legend_elements = [mpl.patches.Patch(facecolor=fc, edgecolor=ec, label=label) for fc, ec, label in zip(facecolors, edgecolors, labels)]
    
    ax.legend(handles=legend_elements,
              ncols=ncols,
              fontsize=fontsize,
              bbox_to_anchor=coords,
              bbox_transform=ax.transAxes,
              loc='center',
              columnspacing=columnspacing)

    return fig
    
    

# RIDGE PLOT
def create_ridgeplot(
    fig:mpl.figure.Figure,
    outer_gs:mpl.gridspec.GridSpec,
    vals: ArrayLike,
    kdes: Dict[str, Callable[...,...]],
    xlimits: Tuple[float],
    ylimits: Tuple[float],
    wspace: float,
    hspace: float,
    order:List[str],
    yaxis_visible:bool,
    xlabel:str,
    xticks:List[float],
    xticklabels:List[str],
    ylabels:Dict[str,str],
    texts:List[str],
    text_coords:List[Tuple[float]],
    text_fontsizes:List[float],
    text_fontweights:List[float],
    text_colors:List[str],
    annotation_list:Optional[Dict[str,...]],
    style:Dict[str,...]
)->mpl.figure.Figure:

    """
    Creates a ridge plot over the values of vals with densities kdes.

    Input
    fig - the figure.
    outer_gs - the gridspec cell to which to add the ridge plot.
    vals - the shared xvalues over which different densities are plotted.
    kdes - dictionary of different densities.
    xlimits - tuple with the xlimits shared across all densities.
    ylimits - tuple with the ylimits shared across all densities.
    wspace - the horizontal spacing, width.
    hspace - the vertical spaceing, height.
    order - list of the keys of kdes that determines the order in which they appear.
    yaxis_visible - whether to keep the yaxis visible
    xlabel - the label of the x-axis
    ylabels - dictionary that replaces the key with a axis label.
    texts - list of text for annotations
    text_coords - list of coordinates for the text
    text_fontsizes - list of font sizes for the text
    text_fontweights - list of font weights for the text
    text_colors - list of colors for the text
    annotation_list - list of dictionaries with the annotations with arrows
    style - dictionary with colors, line widths, etc.

    Output
    fig - the figure
    """

    # extract the x-axis and y-axis limits
    xmin, xmax = xlimits
    ymin, ymax = ylimits

    # count the number of different densities
    num_curves = len(order)

    # make sure vals is a numpy array
    vals = np.asarray(vals)

    # create the inner gridspec for each of the densities
    # there is one column and each row corresponds to a 
    # new density.
    gs = grid_spec.GridSpecFromSubplotSpec(
        nrows=num_curves,
        ncols=1,
        subplot_spec=outer_gs,
        wspace=wspace,
        hspace=hspace
    )

    # populate the gridspec cells with their own axis,
    # and plot the density.
    
    axs = []
    for i in range(0, num_curves):

        # determine the next curve according to the sorting dictionary
        key = order[num_curves - 1 - i]

        # create a new axis
        axs.append(fig.add_subplot(gs[i:i+1, 0]))

        # plot the density
        axs[-1].plot(
            vals,
            kdes[key](vals),
            label=ylabels[key],
            color=style['colors_line'][key],
            alpha=style['alpha_line']
        )
        
        # fill the area under the curve
        axs[-1].fill_between(
            vals,
            0.,
            kdes[key](vals),
            color=style['colors_face'][key],
            alpha=style['alpha_face']
        )

        # add lines that are the zeros for the individual densities
        axs[-1].hlines(0., 
                       xmin, 
                       xmax,
                       color=style['colors_line']['axis'],
                       linewidth=style['linewidth_axis']
                      )

        # set the axis limits
        axs[-1].set_xlim(xmin, xmax)
        axs[-1].set_ylim(ymin, ymax)

        # xticks
        if i==num_curves-1:
            axs[-1].set_xticks(xticks)
            axs[-1].set_xticklabels(xticklabels)
            axs[-1].tick_params(axis='x', labelsize=style['fontsize_xticks'])
            axs[-1].set_xlabel(xlabel, fontsize=style['fontsize_xlabels'])
        else:
            axs[-1].set_xticks([])
            axs[-1].set_xticklabels([])

        # yticks
        if yaxis_visible:
            axs[-1].set_yticks([0])
            axs[-1].set_yticklabels([ylabels[key]])
            axs[-1].tick_params(labelsize=style['fontsize_yticks'])
        else:
            axs[-1].set_yticks([])
            axs[-1].set_yticklabels([])
            
        # remove spines and background
        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            axs[-1].spines[s].set_visible(False)

        axs[-1].patch.set_alpha(0.)

    for annotation_dict in annotation_list:
        
        axs[-1].annotate(
            annotation_dict['text'],
            color=annotation_dict['text_color'],
            fontsize=annotation_dict['text_fontsize'],
            fontstyle=annotation_dict['text_fontstyle'],
            xycoords='axes fraction',
            xy=annotation_dict['arrow_tip'],
            xytext=annotation_dict['arrow_tail'],
            arrowprops=annotation_dict['arrow_props']
        )

    for text, coords, fs, fw, c in zip(texts, text_coords, text_fontsizes, text_fontweights, text_colors):
        x,y = coords
        axs[-1].text(x,y, text, ha='center', transform=axs[0].transAxes, fontsize=fs, fontweight=fw, color=c)
    
    return fig

## VIOLIN PLOT 
def create_violinplot(
    fig:mpl.figure.Figure,
    outer_gs:mpl.gridspec.GridSpec,
    vals:ArrayLike, 
    kdes:Dict[str,Callable[...,...]],
    xlimits:Tuple[float],
    ylimits:Tuple[float],
    hspace:float,
    wspace:float,
    yaxis_visible:bool,
    order:List[str],
    labels:Dict[str,str],
    threshold:float,
    ylabel:str,
    texts:List[str],
    text_coords:List[Tuple[float]],
    text_fontsizes:List[float],
    text_fontweights:List[float],
    text_colors:List[str],
    annotation_list:List[Dict[str,...]],
    style:Dict[str,...]
)->mpl.figure.Figure:
    
    """
    Creates a violin plot over the values of vals with densities kdes.

    Input
    fig - the figure.
    outer_gs - the gridspec cell to which to add the violin plot.
    vals - the values over which different densities are plotted.
    kdes - dictionary of different densities.
    xlimits - tuple with the xlimits shared across all densities.
    ylimits - tuple with the ylimits shared across all densities.
    wspace - the horizontal spacing, width.
    hspace - the vertical spaceing, height.
    yaxis_visible - whether to keep the yaxis visible
    order - list of the keys of kdes that determines the order in which they appear.
    labels - a dictionary that replaces the key with a axis label
    threshold - a threshold below which the density is not plotted
    ylabel - y-axis label
    texts - list of text for annotations
    text_coords - list of coordinates for the text
    text_fontsizes - list of font sizes for the text
    text_fontweights - list of font weights for the text
    text_colors - list of colors for the text
    annotation_list - list of dictionaries with the annotations with arrows
    style - dictionary with colors, line widths, etc.
    annotation_centered - whether the annotation coordinates are with respect to the first (false) or middle (true) violin axis

    Output
    fig - the figure
    """


    # unpack the axis limits
    xmin, xmax = xlimits
    ymin, ymax = ylimits

    # determine the number of violins
    num_curves = len(order)

    # create the gridspec cells corresponding to different
    # violins.
    gs = grid_spec.GridSpecFromSubplotSpec(
        nrows=1,
        ncols=num_curves,
        subplot_spec=outer_gs,
        wspace=wspace,
        hspace=hspace
    )

    # iterate over these cells and create a violin plot
    # at each axis. One half of the violin is on the
    # negative side of this axis the other one on the 
    # positive side.
    axs = []
    for i in range(0, num_curves):

        axs.append(fig.add_subplot(gs[0, i:i+1]))

        # get the key of the density
        key = order[i]

        # plot the left half of the violin. This means everything
        # happens with a negative sign.
        left_x = -kdes[key][0](vals)
        sub_left_x = left_x[np.abs(left_x)>threshold]
        sub_vals_left = vals[np.abs(left_x)>threshold]

        
        axs[-1].plot(
            sub_left_x,
            sub_vals_left,
            color=style['colors_line'][key][0],
            alpha=style['alpha_line']
        )

        axs[-1].fill_betweenx(
            sub_vals_left,
            sub_left_x,
            np.zeros_like(sub_vals_left),
            color=style['colors_face'][key][0],
            alpha=style['alpha_face']
        )

        #  plot the right side of the violin
        right_x = kdes[key][1](vals)
        sub_right_x = right_x[np.abs(right_x)>threshold]
        sub_vals_right = vals[np.abs(right_x)>threshold]
        
        axs[-1].plot(
            sub_right_x,
            sub_vals_right,
            color=style['colors_line'][key][1],
            alpha=style['alpha_line']
        )
        
        axs[-1].fill_betweenx(
            sub_vals_right,
            np.zeros_like(sub_vals_right),
            sub_right_x,
            color=style['colors_face'][key][1],
            alpha=style['alpha_face'])

        # add a central vertical line to the violin
        axs[-1].vlines(
            0,
            np.min([np.min(sub_vals_left), np.min(sub_vals_right)]),
            np.max([np.max(sub_vals_left), np.max(sub_vals_right)]),
            color=style['colors_line']['axis'],
            linewidth=style['linewidth_axis']
        )

        # set axis limits
        axs[-1].set_xlim(-ymax, ymax)
        axs[-1].set_ylim(xmin, xmax)

        # remove tick labels and background patch
        axs[-1].set_xticks([0])                                     # set the xtick
        axs[-1].set_xticklabels([labels[key]])                      # set the xtick label
        axs[-1].tick_params(
            axis='x',
            labelrotation=style['label_rotation'],
            labelsize=style['fontsize_xticks']) 
        axs[-1].patch.set_alpha(0.)                                 # remove background

        # style the y-axis
        if not yaxis_visible or i>0:
            axs[-1].set_yticks([])           # remove yticks
            axs[-1].set_yticklabels([])      # remove ytick labels
            spines = ["top","right","left"]  # remove spines
            for s in spines:
                axs[-1].spines[s].set_visible(False)
        else:
            axs[-1].set_ylabel(ylabel, fontsize=style['fontsize_ylabels'])
            axs[-1].tick_params(axis='y', labelsize=style['fontsize_yticks'])
            spines = ["top","right"]
            for s in spines:
                axs[-1].spines[s].set_visible(False)

    for annotation_dict in annotation_list:
        
        axs[-1].annotate(
            annotation_dict['text'],
            color=annotation_dict['text_color'],
            fontsize=annotation_dict['text_fontsize'],
            fontstyle=annotation_dict['text_fontstyle'],
            xycoords='axes fraction',
            xy=annotation_dict['arrow_tip'],
            xytext=annotation_dict['arrow_tail'],
            arrowprops=annotation_dict['arrow_props']
        )

    for text, coords, fs, fw, c in zip(texts, text_coords, text_fontsizes, text_fontweights, text_colors):
        x,y = coords

        axs[-1].text(x,y, text, ha='center', transform=axs[int(num_curves/2)].transAxes, fontsize=fs, fontweight=fw, color=c)

    return fig

# BAR PLOT
def create_barplot(
    fig:mpl.figure.Figure,
    gs:mpl.gridspec.GridSpec,
    data:Dict[str,float],
    ylimits:Tuple[float],
    norm:float,
    width:float,
    order:List[str],
    yaxis_visible:bool,
    xlabel:str,
    ylabel:str,
    yticks:List[float],
    yticklabels:List[str],
    texts:List[str],
    text_coords:List[Tuple[float]],
    text_fontsizes:List[float],
    text_fontweights:List[float],
    text_colors:List[str],
    annotation_list:List[Dict[str,...]],
    style:Dict[str,...],
)->mpl.figure.Figure:
    
    """
    Creates a bar plot.

    Input
    fig - the figure.
    gs - the gridspec cell to which to add the plot.
    data - the data to be plotted.
    ylimits - tuple with ylimits
    norm - normalization factor
    width - the bar width
    order - order a list of keys to order the data
    yaxis_visible - whether to keep the yaxis visible
    xlabel - x-axis label
    ylabel - y-axis label
    yticks - list of ytick values
    yticklabels - list of ytick labels
    texts - list of text for annotations
    text_coords - list of coordinates for the text
    text_fontsizes - list of font sizes for the text
    text_fontweights - list of font weights for the text
    text_colors - list of colors for the text
    annotation_list - list of dictionaries with the annotations with arrows
    style - dictionary with colors, line widths, etc.

    Output
    fig - the figure
    """
    
    # extract axis limits
    ymin, ymax = ylimits

    # create a new axis
    ax = fig.add_subplot(gs)

    # determine how many cases there are
    num_cases = len(order)

    for i in range(num_cases):

        # get the data for one case
        key = order[i]
        sub_data = data[key]

        # iterate over edge sizes, these correspond to the different
        # clusters of bars. And then stack the contributions of different groups.
        for s in EDGE_SIZES:
            bottom = 0
            for g in GROUPS:
                
                ax.bar(
                    s + i*width,
                    np.mean(sub_data['edge_sizes'][s][g])/norm,
                    bottom=bottom,
                    color=style['colors_face'][key][g],
                    edgecolor=style['colors_line'][key][g],
                    linewidth=style['linewidth'],
                    alpha=style['alpha_line'],
                    width=width)
                
                bottom += np.mean(sub_data['edge_sizes'][s][g])/norm

    # styling of the x-axis and y-axis
    ax.set_xlim(np.min(EDGE_SIZES)-width, np.max(EDGE_SIZES)+3*width)
    ax.set_xticks([s + width for s in EDGE_SIZES])
    ax.set_xticklabels([str(int(s)) for s in EDGE_SIZES])
    ax.tick_params(axis='x', labelsize=style['fontsize_xticks'])
    ax.set_xlabel(xlabel, fontsize=style['fontsize_xlabels'])
    ax.set_ylim(ymin, ymax)
    ax.patch.set_alpha(0.)

    if yaxis_visible:
        ax.set_ylabel(ylabel, fontsize=style['fontsize_ylabels'])
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.tick_params(axis='y', labelsize=style['fontsize_yticks'])
        
        spines = ["top","right"]
        for s in spines:
            ax.spines[s].set_visible(False)
    else:
        ax.set_yticks([])           # remove yticks
        ax.set_yticklabels([])      # remove ytick labels

        spines = ["top","right", "left"]
        for s in spines:
            ax.spines[s].set_visible(False)

    for annotation_dict in annotation_list:
        
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

    for text, coords, fs, fw, c in zip(texts, text_coords, text_fontsizes, text_fontweights, text_colors):
        x,y = coords
        ax.text(x,y, text, ha='center', transform=ax.transAxes, fontsize=fs, fontweight=fw, color=c)
    
    return fig

## ACQUISITION FAIRNESS
def create_acquisition_fairness_plot(
    fig:mpl.figure.Figure,
    gs:mpl.gridspec.GridSpec,
    data:Dict[str,...],
    xlim:Tuple[float],
    ylim:Tuple[float],
    p:float,
    num_iterations:int,
    rng:np.random.Generator,
    yaxis_visible:bool,
    xlabel:str,
    ylabel:str,
    yticks:List[float],
    yticklabels:List[str],
    xticks:List[float],
    xticklabels:List[str],
    texts:List[str],
    text_coords:List[Tuple[float]],
    text_fontsizes:List[float],
    text_fontweights:List[float],
    text_colors:List[str],
    annotation_list:List[Dict[str,...]],
    style:Dict[str,...]
)->mpl.figure.Figure:
    
    """
    Create a plot of acquisition fairness.

    Input
    fig - the figure.
    gs - the gridspec cell to which to add the plot.
    data - the data to be plotted.
    xlim - tuple with xlimits
    ylim - tuple with ylimits
    p - percentile for the confidence interval
    num_iterations - number of iterations for the bootstrap
    yaxis_visible - whether to keep the yaxis visible
    xlabel - x-axis label
    ylabel - y-axis label
    yticks - list of ytick values
    yticklabels - list of ytick labels
    xticks - list of xtick values
    xticklabels - list of xtick labels
    texts - list of text for annotations
    text_coords - list of coordinates for the text
    text_fontsizes - list of font sizes for the text
    text_fontweights - list of font weights for the text
    text_colors - list of colors for the text
    annotation_list - list of dictionaries with the annotations with arrows
    style - dictionary with colors, line widths, etc.

    Output
    fig - the figure
    """

    assert 0 <= p <=100
    
    ax = fig.add_subplot(gs)
    
    for i, key in enumerate(data.keys()):

        # get the number of infected minority nodes
        # at the times when a fraction f of all nodes
        # is informed
        i1f = np.array(data[key]['i1f'])
        num_timeseries = data[key]['num_timeseries']

        # get fractions of informed nodes and 
        fs = data[key]['fs']

        # compute the mean of the acquisition fairness
        alpha_mean = np.array(i1f).mean(axis=0)[fs>0]/fs[fs>0]
        
        # compute bootstrap confidence intervals
        alpha = np.zeros((num_iterations, fs[fs>0].shape[0]))
        
        for i in range(num_iterations):
            bs_idx = rng.integers(0, num_timeseries, num_timeseries)
            alpha[i,:] = i1f[bs_idx,:].mean(axis=0)[fs>0] / fs[fs>0]

        alpha_low = np.percentile(alpha, 100 - p, axis=0)
        alpha_high = np.percentile(alpha, p, axis=0)

        # create the plot
        ax.plot(fs[fs>0], alpha_mean, color=style['colors_line'][key], alpha=style['alpha_line'], lw=style['linewidth'])
        ax.fill_between(fs[fs>0], alpha_low, alpha_high, color=style['colors_fill'][key], alpha=style['alpha_fill'], lw=0)
        ax.hlines(1.0, 0., fs.max(), lw=style['linewidth_axis'], ls=style['linestyle_axis'], color=style['colors_axis'])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, fontsize=style['fontsize_xlabels'])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.tick_params(axis='x', labelsize=style['fontsize_xticks'])

        if yaxis_visible:
            spines = ["top", "right"]
            ax.set_ylabel(ylabel, fontsize=style['fontsize_ylabels'])
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.tick_params(axis='y', labelsize=style['fontsize_yticks'])
        else:
            spines = ["top", "right"]
            ax.set_yticklabels([])
        
        for s in spines:
            ax.spines[s].set_visible(False)

    for annotation_dict in annotation_list:
        
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

    for text, coords, fs, fw, c in zip(texts, text_coords, text_fontsizes, text_fontweights, text_colors):
        x,y = coords
        ax.text(x,y, text, ha='center', transform=ax.transAxes, fontsize=fs, fontweight=fw, color=c)
        
    return fig

## DIFFUSION FAIRNESS
def create_diffusion_fairness_plot(
    fig:mpl.figure.Figure,
    gs:mpl.gridspec.GridSpec,
    result0:Dict[str,...],
    result1:Dict[str,...],
    xlim:Tuple[float],
    ylim:Tuple[float],
    p:float,
    num_iterations:int,
    rng:np.random.Generator,
    yaxis_visible:bool,
    xlabel:str,
    ylabel:str,
    yticks:List[float],
    yticklabels:List[str],
    xticks:List[float],
    xticklabels:List[str],
    texts:List[str],
    text_coords:List[Tuple[float]],
    text_fontsizes:List[float],
    text_fontweights:List[float],
    text_colors:List[str],
    annotation_list:List[Dict[str,...]],
    style:Dict[str,...]
)->mpl.figure.Figure:
    
    """
    Create a plot of diffusion fairness.

    Input
    fig - the figure.
    gs - the gridspec cell to which to add the plot.
    results0 - the data for majority seeding
    results1 - the data for minority seeding
    xlim - tuple with xlimits
    ylim - tuple with ylimits
    p - percentile for the confidence interval
    num_iterations - number of iterations for the bootstrap
    yaxis_visible - whether to keep the yaxis visible
    xlabel - x-axis label
    ylabel - y-axis label
    yticks - list of ytick values
    yticklabels - list of ytick labels
    xticks - list of xtick values
    xticklabels - list of xtick labels
    texts - list of text for annotations
    text_coords - list of coordinates for the text
    text_fontsizes - list of font sizes for the text
    text_fontweights - list of font weights for the text
    text_colors - list of colors for the text
    annotation_list - list of dictionaries with the annotations with arrows
    style - dictionary with colors, line widths, etc.

    Output
    fig - the figure
    """

    assert 0 <= p <=100

    assert sorted(list(result0.keys()))==sorted(list(result1.keys()))
    
    ax = fig.add_subplot(gs)
    
    for i,key in enumerate(result0.keys()):

        if result0[key]['num_timeseries'] != result1[key]['num_timeseries']:
            print("! Warning unequal number of timeseries. Disregarding samples in the larger one.")
        
        num_timeseries = np.min([result0[key]['num_timeseries'], result1[key]['num_timeseries']])
        
        fs = result0[key]['fs']

        # get the times at which a fraction f of the nodes
        # are informed in either of the two seeding conditions
        tf0 = np.array(result0[key]['tfs'])
        tf1 = np.array(result1[key]['tfs'])

        # compute the mean of the diffusion fairness
        try:
            delta_mean = tf0[:num_timeseries, :].mean(axis=0)[fs>0]/tf1[:num_timeseries, :].mean(axis=0)[fs>0]
        except:
            delta_mean = np.ones(fs[fs>0].shape[0])
            temp0 = tf0.mean(axis=0)[fs>0]
            temp1 = tf1.mean(axis=0)[fs>0]
            idx = temp1!=0
            delta_mean[idx] = temp0[idx]/temp1[idx]
            delta_mean[~idx] = np.nan

        # compute bootstrap confidence intervals via bootstrap
        delta = np.zeros((num_iterations, fs[fs>0].shape[0]))
        
        for i in range(num_iterations):
            bs_idx = rng.integers(0, num_timeseries, num_timeseries)
            try:
                delta[i,:] = tf0[bs_idx, :].mean(axis=0)[fs>0] / tf1[bs_idx, :].mean(axis=0)[fs>0]
            except:
                temp0 = tf0[bs_idx, :].mean(axis=0)[fs>0]
                temp1 = tf1[bs_idx, :].mean(axis=0)[fs>0]
                idx = temp1!=0
                delta[i,idx] = temp0[idx]/temp1[idx]
                delta[i,~idx] = np.nan
                              
        delta_low = np.nanpercentile(delta, 100 - p, axis=0)
        delta_high = np.nanpercentile(delta, p, axis=0)

        ax.plot(fs[fs>0], delta_mean, color=style['colors_line'][key], alpha=style['alpha_line'], lw=style['linewidth'])
        ax.fill_between(fs[fs>0], delta_low, delta_high, color=style['colors_fill'][key], alpha=style['alpha_fill'], lw=0)
        ax.hlines(1.0, 0.,fs.max(), lw=style['linewidth_axis'], ls=style['linestyle_axis'], color=style['colors_axis'])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, fontsize=style['fontsize_xlabels'])
        if xticks!=None:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
        ax.tick_params(axis='x', labelsize=style['fontsize_xticks'])

        if yaxis_visible:
            spines = ["top", "right"]
            ax.set_ylabel(ylabel, fontsize=style['fontsize_ylabels'])
            if yticks!=None:
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
            ax.tick_params(axis='y', labelsize=style['fontsize_yticks'])
        else:
            spines = ["top", "right"]
            ax.set_yticklabels([])
        
        for s in spines:
            ax.spines[s].set_visible(False)
            

    for annotation_dict in annotation_list:
        
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

    for text, coords, fs, fw, c in zip(texts, text_coords, text_fontsizes, text_fontweights, text_colors):
        x,y = coords
        ax.text(x,y, text, ha='center', transform=ax.transAxes, fontsize=fs, fontweight=fw, color=c)
        
    return fig


## HOMOPHILY HISTOGRAMS
def create_homophily_barplot(
    fig:mpl.figure.Figure,
    gs:mpl.gridspec.GridSpec,
    h_mean:ArrayLike,
    h_perc:ArrayLike,
    s:int,
    n_hg:int,
    xlabel:str,
    ylabel:str,
    yaxis_visible:bool,
    texts:List[str],
    text_coords:List[Tuple[float]],
    text_fontsizes:List[float],
    text_fontweights:List[float],
    text_colors:List[str],
    style:Dict[str,...]
)->mpl.figure.Figure:
    
    """
    Create a bar plot of homophily values.

    Input
    fig - the figure.
    gs - the gridspec cell to which to add the plot.
    h_mean - the avarage homophily value to be plotted.
    h_perc - the percentiles of homophily values to be plotted as error bars.
    s - the size of the hyperedges
    n_hg - the number of hypergraphs
    xlabel - x-axis label
    ylabel - y-axis label
    texts - list of text for annotations
    text_coords - list of coordinates for the text
    text_fontsizes - list of font sizes for the text
    text_fontweights - list of font weights for the text
    text_colors - list of colors for the text
    annotation_list - list of dictionaries with the annotations with arrows
    style - dictionary with colors, line widths, etc.

    Output
    fig - the figure
    """

    ax = fig.add_subplot(gs)

    barwidth = style['bar_width']
    
    ax.bar(
        np.arange(0, s+1)-barwidth,
        h_mean[0][s],
        width=barwidth,
        align='edge',
        color=style['bar_color'][0],
        edgecolor=style['bar_linecolor'][0],
        linewidth=style['bar_linewidth']
    )

    if n_hg>1:
        ax.errorbar(
            x=np.arange(0, s+1)-barwidth/2,
            y=h_mean[0][s],
            yerr=np.abs(h_perc[0][s] - h_mean[0][s]),
            ls='',
            marker='',
            color=style['bar_color'][0],
            ecolor=style['e_color'],
            capsize=style['e_capsize'],
            elinewidth=style['e_linewidth']
        )
    
    ax.bar(
        np.arange(0, s+1),
        h_mean[1][s],
        width=barwidth,
        align='edge',
        color=style['bar_color'][1],
        edgecolor=style['bar_linecolor'][1],
        linewidth=style['bar_linewidth']
    )
    
    if n_hg > 1:
        ax.errorbar(
            x=np.arange(0, s+1)+barwidth/2,
            y=h_mean[1][s],
            yerr=np.abs(h_perc[1][s] - h_mean[1][s]),
            ls='',
            marker='',
            color=style['bar_color'][1],
            ecolor=style['e_color'],
            capsize=style['e_capsize'],
            elinewidth=style['e_linewidth']
        )
    
    ax.hlines(
        1.0,
        -2*barwidth,
        s+2*barwidth,
        linestyle=style['axis_linestyle'],
        linewidth=style['axis_linewidth'],
        color=style['axis_color']
    )
    
    ax.set_xlim(-0.5, s+0.5)
    ax.set_xticks(np.arange(0,s+1))
    ax.tick_params(axis='x', labelsize=style['fontsize_xticks'])
    ax.tick_params(axis='y', labelsize=style['fontsize_yticks'])

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    if xlabel!=None:
        ax.set_xlabel(xlabel, fontsize=style['fontsize_xlabel'])

    if yaxis_visible:
        ax.set_ylabel(ylabel, rotation=style['label_rotation'])
        ax.yaxis.label.set_position(style['ylabel_coords'])
        ax.set_yticks(style['yticks_if_label'])
        
    for text, coords, fs, fw, c in zip(texts, text_coords, text_fontsizes, text_fontweights, text_colors):
        x,y = coords
        ax.text(x,y, text, ha='center', va='top', transform=ax.transAxes, fontsize=fs, fontweight=fw, color=c)

    return fig

## LOADING OF DATA
def prepare_data_synthetic(directory:str, fs:ArrayLike)->Dict[str,...]:
    """
    Preprocess the simulation results for plotting.

    Input
    directory - directory with simulation results
    fs - list of fractions of informed nodes at which to store values

    Output
    output - dictionary of preprocessed results.
    """

    # load and extract parameters
    params = get_params(directory)
    num_hg = params['num_hg']
    num_seed_cond = params['num_seed_cond']
    seed_strategy = params['seed_strategy']
    num_nodes = params['num_nodes']
    pm = params['p_m']

    # data structures to store results
    emd = []
    tf = []
    tfmax = {g: [] for g in GROUPS}
    i1f = []
    edge_sizes = {s: [[], []] for s in EDGE_SIZES}
    num_timeseries = 0

    # take the 'fraction of nodes' values that are of interest
    # and convert them to number of nodes. Make sure that the 
    # spacing between two of these values is at least one node.
    
    fs = np.asarray(fs)
    assert fs.shape[0] < num_nodes * pm, "minimum fraction is 1/n"
    fmax = np.max(fs)
    f_idx = np.floor(fs * num_nodes).astype(np.int32)

    # iterate over hyprgraphs, and runs and extract the data from file
    for hg_num in range(num_hg):
        for seeding_num in range(num_seed_cond):
            
            # load one simulation output file
            file_path = f'{directory}/results_hg{hg_num}_{seed_strategy}{seeding_num}.pkl'
            results = pd.read_pickle(file_path)

            # aggregate the results from different runs
            for run_num, result in results.items():
                
                # truncate the time series s.t. the maximum number of infected nodes
                # is the same in all runs. This makes sure that the averages are 
                # taken over the same number of samples at all values of f.
                group = result[:, GROUP_COLUMN]

                i0_max = (group==0).shape[0]
                i1_max = (group==1).shape[0]
                i_max = group.shape[0]

                ig_cutoff = (int(fmax * num_nodes * (1. - pm)), int(fmax * num_nodes * pm))
                i_cutoff = int(fmax * num_nodes)

                if ig_cutoff[0] > i0_max or ig_cutoff[1] > i1_max or i_cutoff > i_max:
                    continue
                else:

                    # increasse the counter of eligible time series
                    num_timeseries += 1
                    
                    # rank the nodes by time of infection
                    rank_column = np.arange(i_max).reshape(-1, 1)
                    ranked_result = np.hstack((result, rank_column))

                    # extract the group membership of nodes
                    group = ranked_result[:, GROUP_COLUMN]

                    # determine emd for each run
                    emd.append(wasserstein_distance(ranked_result[group==0, RANK_COLUMN][:ig_cutoff[0]], ranked_result[group==1, RANK_COLUMN][:ig_cutoff[1]]))

                    # determine the times tf at which fraction f of all nodes is informed
                    tf.append(ranked_result[f_idx, TIME_COLUMN])
                    
                    # determine how many minority nodes are infected at times tf
                    i1f.append(np.cumsum(ranked_result[:, GROUP_COLUMN], dtype=np.float32)[f_idx]/(pm * num_nodes))

                    # determine the times at which a fraction fmax of minority or majority nodes is informed
                    idx1 = np.argmin(np.abs(np.cumsum(group) - ig_cutoff[1]))
                    idx0 = np.argmin(np.abs(-np.cumsum(group - 1) - ig_cutoff[0]))

                    tfmax[0].append(ranked_result[idx0, TIME_COLUMN])
                    tfmax[1].append(ranked_result[idx1, TIME_COLUMN])

                    # determine the sizes of edges present in infection events
                    for s in EDGE_SIZES:
                        for g in GROUPS:
                            edge_sizes[s][g].append(np.sum(ranked_result[group==g, EDGE_SIZE_COLUMN]==s))

    
    # create a dictionary that stores the extracted values
    output = {}
    output['num_timeseries'] = num_timeseries
    output['emd'] = emd
    output['edge_sizes'] = edge_sizes
    output['fs'] = fs
    output['tfs'] = tf
    output['tfmax'] = tfmax
    output['i1f'] = i1f

    return output

## COMPUTING PLOT DATA
def compute_kde_ridge(
    data:Dict[str, ...],
    npoints:int=200,
    xmin:Optional[float]=None,
    xmax:Optional[float]=None,
    verbose:bool=False
)->Tuple[Dict[str, Callable[...,...]], Tuple[float], Tuple[float]]:
    
    """
    Compute the kernel density estimator for ridge plots.

    Input
    data - dictionary of preprocessed simulation results
    npoints - the number of x axis points at which to evaluate the density
    xmin - minimum x value
    xmax - maximum x value
    verbose - whether to print the mean of the emd values

    Output
    kde - dictionary of kernel density estimators
    vals - the x values at which the density is evaluated
    xlimits - the x limits
    ylimits - the y limits
    """

    # determine the x-limits
    if xmin == None:
        xmin = np.min([np.min(d['emd']) for d in data.values()])
    if xmax == None:
        xmax = np.max([np.max(d['emd']) for d in data.values()])

    if verbose:
        for key, d in data.items():
            print(f"{key}: {np.mean(d['emd'])}")

    # values at which to evaluate the density
    vals = np.linspace(xmin, xmax, num=npoints)

    # fit kernel density estimator (kde)
    kde = {}
    ymin = 0.
    ymax = 0.
    for key, dat in data.items():
        kde[key] = gaussian_kde(dat['emd'])
        
        ymin = np.min([ymin, np.min(kde[key](vals))])
        ymax = np.max([ymax, np.max(kde[key](vals))])

    return kde, vals, (xmin, xmax), (ymin, ymax)

def compute_kde_violin(
    data:Dict[str, ...],
    npoints:int=200,
    xmin:Optional[float]=None,
    xmax:Optional[float]=None,
    percentile:Optional[float]=None,
)->Tuple[Dict[str, Callable[...,...]], Tuple[float], Tuple[float]]:
    
    """
    Compute the kernel density estimator for violin plots.

    Input
    data - dictionary of preprocessed simulation results
    npoints - the number of x axis points at which to evaluate the density
    xmin - minimum x value
    xmax - maximum x value
    percentile - if not None the xvalues are restricted to this percentile range for the 
                 density evaluation.

    Output
    kde - dictionary of kernel density estimators
    vals - the x values at which the density is evaluated
    xlimits - the x limits
    ylimits - the y limits
    """

    # determine the x boundaries
    if percentile==None:
        if xmin == None:
             xmin = np.min([np.min([np.min(d['tfmax'][g]) for g in [0,1]]) for d in data.values()])
        if xmax == None:
             xmax = np.max([np.max([np.max(d['tfmax'][g]) for g in [0,1]]) for d in data.values()])
    else:
        if xmin == None:
             xmin = np.min([np.min([np.percentile(d['tfmax'][g], 100. - percentile) for g in [0,1]]) for d in data.values()])
        if xmax == None:
             xmax = np.max([np.max([np.percentile(d['tfmax'][g], percentile) for g in [0,1]]) for d in data.values()])
        

    # determine the x values at which to evaluate the density
    vals = np.linspace(xmin, xmax, num=npoints)

    # fit the kde
    kde = {}
    ymin = 0.
    ymax = 0.
    for key, dat in data.items():
        kde[key] = {}
        for g in [0,1]:
            kde[key][g] = gaussian_kde(dat['tfmax'][g])
            ymin = np.min([ymin, np.min(kde[key][g](vals))])
            ymax = np.max([ymax, np.max(kde[key][g](vals))])

                

    return kde, vals, (xmin, xmax), (ymin, ymax)

# DEGREE DISTRIBUTION BINNING
def degree_distribution_binning(
        degree_sequence:ArrayLike,
        nbins:int=10,
        logbinning:bool=True
        )->Tuple[np.ndarray, np.ndarray]:
    """Compute the degree distribution from a degree sequence.
    
    Input
    - degree_sequence : the degree sequence of the network (or several networks concatenated)
    - nbins : the number of bins to use for the histogram
    - logbinning : whether to use logarithmic binning

    Output
    - pk : the degree distribution
    - k  : the bin centeres
    """

    degree_sequence = np.asarray(degree_sequence)

    kmin = np.min(degree_sequence)
    kmax = np.max(degree_sequence)

    # bins
    if logbinning:
        bins = np.logspace(np.log10(kmin), np.log10(kmax+1), base=10, num=nbins)
    else:
        bins = np.linspace(kmin, kmax+1, num=nbins)

    # histogram
    pk, _ = np.histogram(degree_sequence, bins=bins, density=True)
    k = 0.5 * (bins[1:] + bins[:-1])

    return pk, k