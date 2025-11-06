import matplotlib as mpl
import matplotlib.pyplot as plt
from plot_functions import *
import pickle

figsize = (7.2, 8.2)
figure_nrows = 10
figure_ncols = 4
figure_height_ratios = [0.25, 0.65, 0.25, 0.2, 0.65, 0.05, 0.6, 0.05, 0.65, 0.25]
figure_width_ratios = [1, 1, 1, 1]
figure_hspace = 0.75
figure_wspace = 0.65
dpi = 300

hg = 'housebillsgender_genderapi'
figname = f'realworld_SI_{hg}'
stats_dir = './descriptive_stats'
results_dir = './plotdata'
output_dir = './'
spms = [0.0, 1.0]

title_style = {
    'fontsize' : 12,
    'fontcolor' : '#14191a',
    'fontweight' : 'bold'
}

title_coords = (-0.04, 0.4)

groups = [0,1]
edge_sizes = [2,3,4,5]
q_low = 5
q_high = 95

hypergraph_names = {
    'aps_genderapi' : 'APS',
    'aps_genderizerio' : 'APS',
    'dblp_genderapi' : 'DBLP',
    'dblp_genderizerio' : 'DBLP',
    'highschool' : 'High School',
    'hospital' : 'Hospital',
    'housebills' : 'House (Party affiliation)',
    'housebillsgender_genderapi' : 'House',
    'housebillsgender_genderizerio' : 'House',
    'primaryschool' : 'Primary School',
    'senatebills' : 'Senate (Party affiliation)',
    'senatebillsgender_genderapi' : 'Senate',
    'senatebillsgender_genderizerio' : 'Senate'
}

# Bootstrap Params
bootstrap_seed = 213
bootstrap_num = 100
bootstrap_p = 0.90

# General Annotation Params
annotation_fontsizes = {'hom' : [12, 9],
                        'dW'  : [12, 9],
                        't90' : [12, 9],
                        'acq' : [12, 9],
                        'dif' : [12, 9]
                    }

annotation_fontweights = ['bold', 'normal']

annotation_coords = {'hom' : [(-0.15, 1.25), (0.5, 1.25)],
                     'dW'  : [(-0.15, 1.45), (0.5, 1.60)],
                     't90' : [(-2.45, 1.20), (0.0, 1.25)],
                     'acq' : [(-0.15, 1.20), (0.5, 1.25)],
                     'dif' : [(-0.15, 1.20), (0.5, 1.25)]
                    }

annotation_colors = ['k', '#272c2d']
small_annotation_color = '#525758'

abc = {
    (1,0) : "(a)",
    (1,1) : "(b)",
    (1,2) : "(c)",
    (1,3) : "(d)",
    (3,0) : "(e)",
    (3,1) : "(f)",
    (3,2) : "(g)",
    (5,0) : "(h)",
    (5,1) : "(i)",
    (5,2) : "(j)",
    (4,3) : "(k)",
}

s_dict = {
    2 : r'$s = 2$',
    3 : r'$s = 3$',
    4 : r'$s = 4$',
    5 : r'$s = 5$'
}


# used order of plots (in ridge and violin)
dynamics_order = [
    'linear',
    'sublinear',
    'superlinear',
    'asymmetric'
]

dynamics_labels = {
    'linear'      : 'lin.',
    'sublinear'   : 'sub.',
    'superlinear' : 'sup.',
    'asymmetric'  : 'asym.'
}

# Seeding strategy annotations
seeding_text_coords_1 = (-0.04, 1.32)
seeding_text_coords_2 = (-0.04, 0.40)
seeding_text_style = {
    'fontsize' : 9,
    'fontcolor' : annotation_colors[-1],
    'fontweight' : 'bold'
}

# Homophily Barplot Parameters
hom_text_coords = (-0.04, -0.15)
hom_text_style = {
    'fontsize' : annotation_fontsizes['hom'][-1],
    'fontcolor' : annotation_colors[-1],
    'fontweight' : annotation_fontweights[-1]
}

hom_xlabel = r'$r$'
hom_ylabel = {2 : r'$h^{(g)}_{2,r}$', 3 : r'$h^{(g)}_{3,r}$', 4 : r'$h^{(g)}_{4,r}$', 5 : r'$h^{(g)}_{5,r}$'}
hom_yticks = {
    'highschool': {2 : [0, 1], 3 : [0, 1, 2], 4 : [0, 1, 2, 3, 4], 5 : [0, 1, 2]},
    'housebillsgender_genderapi' : {2 : [0, 1], 3 : [0, 1, 2], 4 : [0, 1, 3, 6], 5 : [0, 25, 50]}
}
hom_yaxis_visible = True

hom_style = {
    'e_color' : '#474747',
    'e_capsize' : 2,
    'e_linewidth' : 1.5,
    'axis_linestyle' : '--',
    'axis_linewidth' : 1,
    'axis_color' : '#363737',
    'fontsize_xlabel' : 9,
    'fontsize_ylabel' : 9,
    'fontsize_xticks' : 8,
    'fontsize_yticks' : 8,
    'bar_color' : ('#c2d58a','#6d71b7'),
    'bar_linecolor' : ('#a6b96f', '#53589c'),
    'bar_linewidth' : 0.75,
    'bar_width' : 0.25,
    'label_rotation' : 90,
    'ylabel_coords' : (-0.65, 0.5),
    'yticks_if_label' : None # is set in dependence of s above (hom_yticks)
}

# Homophily Legend
hom_legend_labels = [
        r'majority$\,g=0$',
        r'minority$\,g=1$',
]

hom_legend_linecolors = [
        hom_style['bar_linecolor'][0],
        hom_style['bar_linecolor'][1]
]

hom_legend_facecolors = [
        hom_style['bar_color'][0],
        hom_style['bar_color'][1]
]

hom_legend_fontsize = 7
hom_legend_ncols = 2
hom_legend_coords = (0.5, 0.25)
hom_legend_columnspacing = 2.0

# Wasserstein Ridge Plot Parameters
wasserstein_ridge_xmin = 0.
wasserstein_ridge_xmax = lambda n_nodes, hg: n_nodes/4. if 'aps' in hg or 'dblp' in hg else n_nodes/2.
wasserstein_ridge_ymin = 0.
wasserstein_ridge_ymax = None
wasserstein_ridge_npoints = 200

wasserstein_ridge_hspace = -0.5
wasserstein_ridge_wspace = 0.0
wasserstein_ridge_order = dynamics_order
wasserstein_ridge_labels = dynamics_labels
wasserstein_ridge_xlabel = r'$d_W(\mathcal{Z}^{(0)},\mathcal{Z}^{(1)})$'
wasserstein_yaxis_visible = True

wasserstein_ridge_xticks = lambda n_nodes, hg: [0.0, n_nodes/8., n_nodes/4.] if 'aps' in hg or 'dblp' in hg else [0.0, n_nodes/4., n_nodes/2.]
wasserstein_ridge_xticklabels = lambda n_nodes, hg: [r'$0.0$', r'$n/8$', r'$n/4$'] if 'aps' in hg or 'dblp' in hg else [r'$0.0$', r'$n/4$', r'$n/2$']

wasserstein_ridge_style = {
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
    'fontsize_xlabels' : 9,
    'fontsize_xticks'  : 8,
    'fontsize_yticks' : 8,
}

wasserstein_ridge_annotations = [
    {
    'text' : '',
    'text_fontsize': 6,
    'text_color' : small_annotation_color,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (-0.05, 2.05),
    'arrow_tail' : (0.20, 2.05), 
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
    'arrow_tip' :  (-0.05, 2.25), 
    'arrow_tail' : (-0.05, 2.25), 
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
    'arrow_tip' :  (0.95, 2.05), 
    'arrow_tail' : (0.75, 2.05),
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
    'arrow_tip' :  (0.75, 2.25),
    'arrow_tail' : (0.75, 2.25),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.0}
    }
]

# t90 violing plot parameters
violin_hspace = 0.
violin_wspace = 0.
violin_order = dynamics_order
violin_labels = dynamics_labels
violin_ylabel = r'$t^{(g)}_{90}$'

violin_xmin = 0.
violin_xmax = None
violin_ymin = 0.
violin_ymax = None
violin_npoints = 400

violin_percentile =  lambda hg : 85 if 'billsgender' in hg else 99
violin_threshold = 0
violin_yaxis_visible = True

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
    'fontsize_xlabels' : 9,
    'fontsize_ylabels' : 9,
    'fontsize_xticks'  : 8,
    'fontsize_yticks' : 8,
    'label_rotation' : 40 
}

violin_annotations = [
    {
    'text' : 'maj.',
    'text_color' : small_annotation_color,
    'text_fontsize': 6,
    'text_fontstyle': 'italic',
    'arrow_tip' :  (-0.35, 0.88),
    'arrow_tail' : (-0.35, 0.88),
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
    'arrow_tip' :  (0.65, 0.88),
    'arrow_tail' : (0.65, 0.88),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.0}
    }
]

# Acquisition Fairness Parameters
acquisition_xlabel = r'$f$'
acquisition_ylabel = r'$\alpha(f)$'
acquisition_ylim = (0., 1.6)
acquisition_yticks = [0.0, 0.5, 1.0, 1.5]
acquisition_yticklabels = ['0.0', '0.5', '1.0', '1.5']
acquisition_xlim = (0.0, 0.9)
acquisition_xticks = [0.0, 0.3, 0.6, 0.9]
acquisition_xticklabels = ['0.0', '0.3', '0.6', '0.9']

acquisition_yaxis_visible = True

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
    'fontsize_xlabels' : 9,
    'fontsize_ylabels' : 9,
    'fontsize_xticks'  : 8,
    'fontsize_yticks' : 8,
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
    'arrow_tip' :  (0.35, 0.85),
    'arrow_tail' : (0.35, 0.85),
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
    'arrow_tip' :  (0.35, 0.12),
    'arrow_tail' : (0.35, 0.12),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.0}
    }
]

# Diffusion Fairness Parameters
diffusion_xlabel = r'$f$'
diffusion_ylabel = r'$\delta(f)$'
diffusion_yticks = None
diffusion_yticklabels = None
diffusion_xlim = (0., 0.9)
diffusion_xticks = [0.0, 0.3, 0.6, 0.9]
diffusion_xticklabels = ['0.0', '0.3', '0.6', '0.9']
diffusion_yaxis_visible = True

diffusion_style = {
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
    'alpha_fill' : 0.5,
    'linewidth' : 1,
    'linewidth_axis' : 1,
    'linestyle_axis' : '--',
    'fontsize_xlabels' : 9,
    'fontsize_ylabels' : 9,
    'fontsize_xticks'  : 8,
    'fontsize_yticks' : 8,
    'label_rotation' : 40 
}


diffusion_ylim = {
    'aps_genderapi' : (0., 1.5),
    'aps_genderizerio' : (0., 1.5),
    'dblp_genderapi' : (0., 1.5),
    'dblp_genderizerio' : (0., 1.5),
    'highschool' : (0., 1.5),
    'hospital' : (0., 1.5),
    'housebills' : (0., 5.0),
    'housebillsgender_genderapi' : (-20., 35),
    'housebillsgender_genderizerio' : (0., 35),
    'primaryschool' : (0., 1.5),
    'senatebills' : (0., 5.0),
    'senatebillsgender_genderapi' : (0., 1.5),
    'senatebillsgender_genderizerio' : (0., 1.5)
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
    'arrow_tip' :  (0.35, 0.80),
    'arrow_tail' : (0.35, 0.80),
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
    'arrow_tip' :  (0.9, 0.10),
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
    'arrow_tip' :  (0.35, 0.18),
    'arrow_tail' : (0.35, 0.18),
    'arrow_props' :{
        'arrowstyle':'->',
        'color':'#586170',
        'lw':0.0}
    }
]

## Dynamics Legend
dynamics_legend_labels = [
        'linear',
        'sublinear',
        'superlinear',
        'asymmetric'
]

dynamics_legend_linecolors = [
        violin_style['colors_line']['linear'][0],
        violin_style['colors_line']['sublinear'][0],
        violin_style['colors_line']['superlinear'][0],
        violin_style['colors_line']['asymmetric'][0]
]

dynamics_legend_facecolors = [
        violin_style['colors_face']['linear'][0],
        violin_style['colors_face']['sublinear'][0],
        violin_style['colors_face']['superlinear'][0],
        violin_style['colors_face']['asymmetric'][0]
]

dynamics_legend_fontsize = 7
dynamics_legend_ncols = 4
dynamics_legend_columnspacing = 2.0
dynamics_legend_coords = (0.5, 0.05)


## Seed the Random Number Generator
rng = np.random.default_rng(seed=bootstrap_seed)

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

## Add the name of the hypergraph
fig = add_text(fig, gs[0,:], hypergraph_names[hg], title_coords, title_style)


## Add the homophily plots ##

# load homophily data
with open(f"{stats_dir}/{hg}.pkl", "rb") as f:
    
    stats_dict = pickle.load(f)

# compute homophily
n_nodes = stats_dict['num_nodes'][0][-1]
n_hg = stats_dict['num_nodes'].shape[0]

h_mean = {g : {s : np.nanmean(stats_dict['homophily'][g][s], axis=0) for s in edge_sizes} for g in groups}
h_perc = {g : {s : np.nanpercentile(stats_dict['homophily'][g][s], [q_low, q_high], axis=0) for s in edge_sizes} for g in groups}

# add homophily label
fig = add_text(fig, gs[0, :], "Homophily", hom_text_coords, hom_text_style)

# plot homophily
for j, s in enumerate(edge_sizes):

    hom_style['yticks_if_label'] = hom_yticks[hg][s]

    fig=create_homophily_barplot(
        fig,
        gs[1,j],
        h_mean,
        h_perc,
        s,
        n_hg,
        hom_xlabel,
        hom_ylabel[s],
        hom_yaxis_visible,
        [abc[(1,j)], s_dict[s]],
        annotation_coords['hom'],
        annotation_fontsizes['hom'],
        annotation_fontweights,
        annotation_colors,
        hom_style
        )

# add the legend for homophily
fig = add_legend(
    fig,
    gs[2,:],
    hom_legend_facecolors,
    hom_legend_linecolors,
    hom_legend_labels,
    hom_legend_ncols,
    hom_legend_columnspacing,
    hom_legend_coords,
    hom_legend_fontsize,
)

## Load the data for the remaining plots
results = {}
with open(f'{results_dir}/{hg}_plotdata.pkl', 'rb') as f:

    hg_result = pickle.load(f)

for key, val in hg_result.items():

    key = (key[0], key[1], hg)

    results[key] = val


# add the seeding annotation
fig = add_text(fig, gs[3, :], "Majority seeding", seeding_text_coords_1, seeding_text_style)

## Add the Wasserstein Distance Plot
sub_result = {key[1] : val for key, val in results.items() if key[0]==spms[0] and key[2]==hg}

# compute the ridges
ridge_kdes, ridge_vals, ridge_xlimits, r_ylimits = compute_kde_ridge(
        sub_result,
        npoints=wasserstein_ridge_npoints,
        xmin=wasserstein_ridge_xmin,
        xmax=wasserstein_ridge_xmax(n_nodes=n_nodes, hg=hg)
)

# update axis limits for ridge plot
ridge_ylimits = [0.,0.]
if wasserstein_ridge_ymin:
    ridge_ylimits[0] = wasserstein_ridge_ymin
else:
    ridge_ylimits[0] = r_ylimits[0]
if wasserstein_ridge_ymax:
    ridge_ylimits[1] = wasserstein_ridge_ymax
else:
    ridge_ylimits[1] = r_ylimits[1]
ridge_ylimits = tuple(ridge_ylimits)


# create the ridge plot
fig = create_ridgeplot(
    fig,
    gs[4:6,0],
    ridge_vals,
    ridge_kdes,
    ridge_xlimits,
    ridge_ylimits,
    wasserstein_ridge_wspace,
    wasserstein_ridge_hspace,
    wasserstein_ridge_order,
    wasserstein_yaxis_visible,
    wasserstein_ridge_xlabel,
    wasserstein_ridge_xticks(n_nodes=n_nodes, hg=hg),
    wasserstein_ridge_xticklabels(n_nodes=n_nodes, hg=hg),
    wasserstein_ridge_labels,
    [abc[(3,0)], "Inequality in rank\ndistributions"],
    annotation_coords['dW'],
    annotation_fontsizes['dW'],
    annotation_fontweights,
    annotation_colors,
    wasserstein_ridge_annotations,
    wasserstein_ridge_style
)


## Add the t90 Plots
sub_result = {key[1] : val for key, val in results.items() if key[0]==spms[0] and key[2]==hg}

violin_kdes, violin_vals, violin_xlimits, v_ylimits = compute_kde_violin(
        sub_result,
        npoints=violin_npoints,
        xmin=violin_xmin,
        xmax=violin_xmax,
        percentile=violin_percentile(hg)
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
    gs[4:6, 1],
    violin_vals, 
    violin_kdes,
    violin_xlimits,
    violin_ylimits,
    violin_hspace,
    violin_wspace,
    violin_yaxis_visible,
    violin_order,
    violin_labels,
    violin_threshold,
    violin_ylabel,
    [abc[(3,1)], "Time to\nreach 90%"],
    annotation_coords['t90'],
    annotation_fontsizes['t90'],
    annotation_fontweights,
    annotation_colors,
    violin_annotations,
    violin_style
)


## Add the Acquisition Fairness Plots
sub_results = {key[1] : val for key, val in results.items() if key[0]==spms[0] and key[2]==hg}

create_acquisition_fairness_plot(
    fig,
    gs[4:6,2],
    sub_results,
    acquisition_xlim,
    acquisition_ylim,
    bootstrap_p,
    bootstrap_num,
    rng,
    acquisition_yaxis_visible,
    acquisition_xlabel,
    acquisition_ylabel,
    acquisition_yticks,
    acquisition_yticklabels,
    acquisition_xticks,
    acquisition_xticklabels,
    [abc[(3,2)], "Acquisition\nfairness"],
    annotation_coords['acq'],
    annotation_fontsizes['acq'],
    annotation_fontweights,
    annotation_colors,
    acquisition_annotations,
    acquisition_style
)

# add the seeding annotation
fig = add_text(fig, gs[6,:], "Minority seeding", seeding_text_coords_2, seeding_text_style)

## Add the Wasserstein Distance Plot
sub_result = {key[1] : val for key, val in results.items() if key[0]==spms[0] and key[2]==hg}

# compute the ridges
ridge_kdes, ridge_vals, ridge_xlimits, r_ylimits = compute_kde_ridge(
        sub_result,
        npoints=wasserstein_ridge_npoints,
        xmin=wasserstein_ridge_xmin,
        xmax=wasserstein_ridge_xmax(n_nodes=n_nodes, hg=hg)
)

# update axis limits for ridge plot
ridge_ylimits = [0.,0.]
if wasserstein_ridge_ymin:
    ridge_ylimits[0] = wasserstein_ridge_ymin
else:
    ridge_ylimits[0] = r_ylimits[0]
if wasserstein_ridge_ymax:
    ridge_ylimits[1] = wasserstein_ridge_ymax
else:
    ridge_ylimits[1] = r_ylimits[1]
ridge_ylimits = tuple(ridge_ylimits)


# create the ridge plot
fig = create_ridgeplot(
    fig,
    gs[7:9,0],
    ridge_vals,
    ridge_kdes,
    ridge_xlimits,
    ridge_ylimits,
    wasserstein_ridge_wspace,
    wasserstein_ridge_hspace,
    wasserstein_ridge_order,
    wasserstein_yaxis_visible,
    wasserstein_ridge_xlabel,
    wasserstein_ridge_xticks(n_nodes=n_nodes, hg=hg),
    wasserstein_ridge_xticklabels(n_nodes=n_nodes, hg=hg),
    wasserstein_ridge_labels,
    [abc[(5,0)], "Inequality in rank\ndistribution"],
    annotation_coords['dW'],
    annotation_fontsizes['dW'],
    annotation_fontweights,
    annotation_colors,
    [],
    wasserstein_ridge_style
)


## Add the t90 Plots
sub_result = {key[1] : val for key, val in results.items() if key[0]==spms[1] and key[2]==hg}

violin_kdes, violin_vals, violin_xlimits, v_ylimits = compute_kde_violin(
        sub_result,
        npoints=violin_npoints,
        xmin=violin_xmin,
        xmax=violin_xmax,
        percentile=violin_percentile(hg)
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
    gs[7:9, 1],
    violin_vals, 
    violin_kdes,
    violin_xlimits,
    violin_ylimits,
    violin_hspace,
    violin_wspace,
    violin_yaxis_visible,
    violin_order,
    violin_labels,
    violin_threshold,
    violin_ylabel,
    [abc[(5,1)], "Time to\nreach 90%"],
    annotation_coords['t90'],
    annotation_fontsizes['t90'],
    annotation_fontweights,
    annotation_colors,
    [],
    violin_style
)


## Add the Acquisition Fairness Plots
sub_results = {key[1] : val for key, val in results.items() if key[0]==spms[1] and key[2]==hg}

create_acquisition_fairness_plot(
    fig,
    gs[7:9,2],
    sub_results,
    acquisition_xlim,
    acquisition_ylim,
    bootstrap_p,
    bootstrap_num,
    rng,
    acquisition_yaxis_visible,
    acquisition_xlabel,
    acquisition_ylabel,
    acquisition_yticks,
    acquisition_yticklabels,
    acquisition_xticks,
    acquisition_xticklabels,
    [abc[(5,2)], "Acquisition\nfairness"],
    annotation_coords['acq'],
    annotation_fontsizes['acq'],
    annotation_fontweights,
    annotation_colors,
    [],
    acquisition_style
)

## Add the Diffusion Fairness Plots
sub_results_0 = {key[1] : val for key, val in results.items() if key[0]==0}
sub_results_1 = {key[1] : val for key, val in results.items() if key[0]==1}

create_diffusion_fairness_plot(
        fig,
        gs[5:7,3],
        sub_results_0,
        sub_results_1,
        diffusion_xlim,
        diffusion_ylim[hg],
        bootstrap_p,
        bootstrap_num,
        rng,
        diffusion_yaxis_visible,
        diffusion_xlabel,
        diffusion_ylabel,
        diffusion_yticks,
        diffusion_yticklabels,
        diffusion_xticks,
        diffusion_xticklabels,
        [abc[(4,3)], "Diffusion\nfairness"],
        annotation_coords['dif'],
        annotation_fontsizes['dif'],
        annotation_fontweights,
        annotation_colors,
        diffusion_annotations,
        diffusion_style
    )


## create the legend
fig = add_legend(
    fig,
    gs[9,:],
    dynamics_legend_facecolors,
    dynamics_legend_linecolors,
    dynamics_legend_labels,
    dynamics_legend_ncols,
    dynamics_legend_columnspacing,
    dynamics_legend_coords,
    dynamics_legend_fontsize,
)

# save figure
fig.savefig(output_dir +  figname + '.svg',  bbox_inches='tight')
fig.savefig(output_dir +  figname + '.pdf',  bbox_inches='tight')
fig.savefig(output_dir +  figname + '.png', dpi=dpi,  bbox_inches='tight')