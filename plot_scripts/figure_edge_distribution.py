"""
Figure: Edge Distributions

This script generates plots for the edge
size distributions of the real world
hypergraphs.

SD - 2025/05/16
"""

### IMPORT ###
import matplotlib as mpl
import numpy as np
import pickle as pkl
import sys
import os

from plot_functions import create_figure, add_text

### PARAMETERS ###

# Paths
base_path = './hypergraphs/'
output_path = './'
figname = 'edgesize_distributions'

# Datasets
names = [
    'primaryschool',
    'highschool',
    'hospital',
    'housebills',
    'aps',
    'senatebills',
    'dblp'
]
titles = [
    'Primary School',
    'High School',
    'Hospital',
    'House',
    'APS',
    'Senate',
    'DBLP'
]

# Labels
abc = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
log_bin_names = ['housebills', 'aps', 'senatebills', 'dblp']

# Coordinates
coordinates = [
    (1, 0),  # Primary School
    (1, 1),  # High School
    (1, 2),  # Hospital
    (3, 0),  # House
    (3, 1),  # APS
    (3, 2),  # Senate
    (5, 0),  # DBLP
]

# Figure settings
figsize = (7.2, 7.2)  # inches
figure_hspace = 0.75
figure_wspace = 0.25
figure_ncols = 3
figure_nrows = 8
figure_height_ratios = [0.1, 1.0, 0.15, 1.0, 0.15, 1.0, 0.05, 0.05]
figure_width_ratios = [1., 1., 1.]
dpi = 500

# Binning parameters
bin_width = 1
density = True

# Axis labels
xlabel = 'Edge Size'
ylabel = 'Density'

ylabel_fontsize = 10
xlabel_fontsize = 10
ticklabelsize = 8

# Bar styling
bar_facecolor = '#287c3720'
bar_edgecolor = '#287c37BF'
e_linewidth = 0.8

# Header
header = 'Edge Size Distributions in Real-world Hypergraphs'
header_coords = (-0.03, 0.5)
header_style = {
    'fontsize': 10,
    'fontcolor': '#14191a',
    'fontweight': 'bold'
}

# Small annotation settings
text_fontsizes = [10, 8]
text_fontweights = ['bold', 'normal']
text_coords = [(-0.05, 1.15), (0.5, 1.15)]
text_colors = ['k', '#272c2d']


### Load_hypergraphs ###

def load_hgs():
    edge_sizes = {}
    for name in names:
        try:
            if name in ['dblp', 'aps']:
                fp = f'{base_path}/{name}/hypergraphs_genderapi/hg_lcc_0.pkl'
            else:
                fp = f'{base_path}/{name}/hypergraphs/{name}_lcc_hg.pkl'

            if not os.path.exists(fp):
                raise FileNotFoundError(f'Missing file: {fp}')

            with open(fp, 'rb') as f:
                hg = pkl.load(f)

            edges = list(hg.edge_size.values())
            edge_sizes[name] = edges

        except FileNotFoundError as e:
            print(f'[{name}] File not found: {e}')
        except Exception as e:
            print(f'[{name}] Failed to load data: {e}')

    return edge_sizes

### MAIN ###

# Check for data
if base_path is None:
    sys.exit('Error: base_path is not set. Please make sure the data is prepared and set it to the root directory '
             'containing the data folders.')

# Load data
edge_sizes_dict = load_hgs()
if not edge_sizes_dict:
    sys.exit('Error: No data loaded. Check that your base_path is correct and data files exist.')

# Create the figure
fig, gs = create_figure(
    figsize=figsize,
    nrows=figure_nrows,
    ncols=figure_ncols,
    height_ratios=figure_height_ratios,
    width_ratios=figure_width_ratios,
    hspace=figure_hspace,
    wspace=figure_wspace
)

# Add header
fig = add_text(fig, gs[0, :], header, header_coords, header_style)

# Determine the maximum edge size across all graphs for consistent x-axis
all_edge_sizes = np.concatenate(list(edge_sizes_dict.values()))
max_edge_size = int(np.max(all_edge_sizes))
bins = np.arange(1, max_edge_size + 2) - 0.5  # bin from 0.5 to (max + 0.5)

# Plot histograms
for i, name in enumerate(names):
    ax = fig.add_subplot(gs[coordinates[i]])

    sizes = np.array(edge_sizes_dict[name])

    # Check if this hypergraph should be log-binned
    if name in log_bin_names:
        # Log binning
        min_size = max(1, sizes.min())
        max_size = sizes.max()
        bins = np.logspace(np.log10(min_size), np.log10(max_size), num=15)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())

    else:
        # Linear binning
        min_size = sizes.min()
        max_size = sizes.max()
        bins = np.arange(min_size - 0.5, max_size + 1.5, 1)  # center on integers
        if name not in log_bin_names:
            ax.set_xticks(np.arange(2, 6))  # ticks at 2,3,4,5
            ax.set_xticklabels(['2', '3', '4', '5'])  # explicit labels

    ax.hist(
        sizes,
        bins=bins,
        density=True,
        color=bar_facecolor,
        edgecolor=bar_edgecolor,
        linewidth=e_linewidth
    )

    # X-axis styling
    ax.tick_params(axis='x', labelsize=ticklabelsize)
    ax.set_xlabel(xlabel, fontsize=ticklabelsize)

    # Y-axis styling
    if coordinates[i][1] == 0:
        ax.set_ylabel(ylabel, fontsize=ticklabelsize)

        # Special adjustment for plot (g)
        if i == 0:
            ax.yaxis.set_label_coords(-0.25, 0.5)

    ax.tick_params(axis='x', labelsize=ticklabelsize)
    ax.tick_params(axis='y', labelsize=ticklabelsize)

    # remove top/right spines
    spines = ['top', 'right']
    for s in spines:
        ax.spines[s].set_visible(False)

    ax.patch.set_alpha(0.)

    # Add small annotations (a, b, c, ..., and dataset title)
    for text, coords, fs, fw, c in zip([abc[i], titles[i]], text_coords, text_fontsizes, text_fontweights,
                                       text_colors):
        x, y = coords
        ax.text(x, y, text, ha='center', transform=ax.transAxes, fontsize=fs, fontweight=fw, color=c)

# Save
fig.savefig(output_path + figname + '.svg', bbox_inches='tight')
fig.savefig(output_path + figname + '.pdf', bbox_inches='tight')
fig.savefig(output_path + figname + '.png', dpi=dpi, bbox_inches='tight')