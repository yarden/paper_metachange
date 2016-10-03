##
## Make suddenswitch paper
##
## Figure dimensions
# widths for figures:
# 1 column, 85 mm = 3.3 in
# 1.5 column, 114 mm = 4.48
# 2 column, 174 mm (the full width of the page) = 6.8
import os
import sys
import copy
import time
import cPickle as pickle
from collections import OrderedDict

import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style("ticks")

import numpy as np
import pandas

import particlefever
import particlefever.switch_ssm as switch_ssm
import particlefever.particle_filter as particle_filter
import particlefever.sampler as sampler

import paths
import time_unit
import simulation
import fitness
import growth_rates
import model_sudden_markov
import model_switch_ssm
import model_flat_bayes
import pipeline_sudden
import plot_utils
import prob_utils
import utils
import sbml

import shutil

from ruffus import *

from pipeline_sudden import *

##
## default parameter filenames
##
FIGS_DIR = os.path.join(paths.PLOTS_DIR, "paper_figures")
utils.make_dir(FIGS_DIR)

##
## figure settings
##
# font size for panel label ("A", "B", ...)
PANEL_LABEL_FONTSIZE = 14

# emtpy dummy file that starts the whole paper pipeline
PAPER_PIPELINE_START = \
  os.path.join(paths.PIPELINES_INFO, "paper_pipeline.start")
  
##
## SBML model for figure6
##
SBML_MODEL_FNAME = \
  os.path.join(paths.MAIN_DIR,
               "sbml_models",
               "glu_gal_transition_counter.xml")

##
## utilities for making figures
##
def save_figure(plot_fname, fig_name):
    """
    Take a given figure filename and copy it
    to the appropriate figure subdirectory.
    """
    fig_dir = os.path.join(FIGS_DIR, fig_name)
    utils.make_dir(fig_dir)
    base_name = os.path.basename(plot_fname)
    print "saving %s to %s" %(plot_fname, fig_dir)
    shutil.copyfile(plot_fname, os.path.join(fig_dir, base_name))

@originate(paths.PIPELINE_START_FNAME)
def run_analyses(start_fname):
    print "running all analyses for paper"
    print "-----"
    with open(start_fname, "w") as out_file:
        out_file.write("started paper pipeline.\n")
    # call all pipelines here
    pipeline_sudden.start_pipeline(paths.PIPELINE_SUDDEN_START_FNAME)

@files(run_analyses,
       os.path.join(FIGS_DIR, "figure1.pdf"),
       "figure1")
def make_figure1(start_fname, plot_fname, label,
                 panel_label_x=0.01):
    """
    Figure 1 is a schematic made in Illustrator.
    """
    sns.set_style("ticks")
    fig = plt.figure(figsize=(3, 4.5))
    # make suddenswitch environment schematic
    num_plots = 4
    num_steps = 10
    time_axis = range(11)
    time_min = 0
    time_max = 10
    labels_fontsize = 8
    gs = gridspec.GridSpec(num_plots, 1)
    gs.update(hspace=0.71,
              top=0.94,
              bottom=0.38,
              left=0.20,
              right=0.91)
    ## nutrient levels
    ax1 = plt.subplot(gs[0, 0])
    c = 0.05
    # coordinates of y-label (used to align ylabels
    # across subplots)
    #ylabel_x = -0.3
    ylabel_x = -0.15
    ylabel_y = 0.5
    nutr1_x = [0, 2, 4, 6, 8, 10]
    nutr1_y = [1, 1, c, 1, c, 1]
    nutr2_x = [0, 2.5, 4.5, 6.5, 8.5, 10]
    nutr2_x = np.array(nutr2_x) + c*4
    nutr2_y = [c, c, 1, c, 1, c]
    plt.step(nutr1_x, nutr1_y, label="Nutr. 1",
             linestyle="-",
             clip_on=False,
             color=plot_utils.lightgreen)
    line = plt.step(nutr2_x, nutr2_y, label="Nutr. 2",
                    linestyle="--",
                    clip_on=False,
                    color=plot_utils.blue)
    line[0].set_dashes((3, 4))
    plt.ylim(0, 1 + c)
    plt.xlim(0 - c, nutr1_x[-1] + c)
    plt.xticks(time_axis)
    plt.yticks(np.arange(0, 1.5, 0.5), fontsize=8)
    plt.ylabel("Nutr. level", fontsize=labels_fontsize)
    from matplotlib.ticker import FormatStrFormatter
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    box = ax1.get_position()
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels,
               bbox_to_anchor=(0.27, 1.48),
               handlelength=2.0,
               loc=2,
               ncol=2,
               borderaxespad=0,
               fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8,
                    pad=2)
    ax1.yaxis.set_label_coords(ylabel_x, ylabel_y)
    plt.annotate("A", xy=(panel_label_x, 0.965),
                 xycoords="figure fraction",
                 fontsize=PANEL_LABEL_FONTSIZE,
                 fontweight="bold")
    ## draw line to indicate gut-related changes
    plt.annotate(r"Gut-related changes", xy=(panel_label_x, 0.68),
                 xycoords="figure fraction",
                 rotation=90,
                 fontsize=8)
    ax1.annotate('', xy=(-0.30, -0.5),
                 xycoords='axes fraction',
                 xytext=(-0.30, -1.0), 
                 arrowprops=dict(arrowstyle="->", color='k'))
    ## temperature
    ax2 = plt.subplot(gs[1, 0])
    plt.ylabel("Temp. (C)", fontsize=labels_fontsize)
    temp_x_axis = [0, 2, 5, 10]
    temp_y_axis = [25, 25, 27, 37]
    plt.step(temp_x_axis, temp_y_axis, color="k")
    plt.xlim(0 - c, nutr1_x[-1] + c)
    plt.xticks(time_axis)
    plt.yticks(range(24, 38 + 8, 8))
    plt.tick_params(axis='both', which='major', labelsize=8,
                    pad=2)
    ax2.yaxis.set_label_coords(ylabel_x, ylabel_y)
    ## pH
    ax3 = plt.subplot(gs[2, 0])
    plt.ylabel("pH", fontsize=labels_fontsize)
    pH_x = [0, 1, 3, 5, 7, 9, 10]
    pH_y = [1.5, 1.5, 6.4, 7.4, 7.4, 5.9, 6.5] 
    plt.step(pH_x, pH_y, color="k", clip_on=False)
    plt.xlim(0 - c, nutr1_x[-1] + c)
    plt.xticks(time_axis)
    plt.yticks(range(1, 11, 4))
    plt.tick_params(axis='both', which='major', labelsize=8,
                    pad=2)
    ax3.yaxis.set_label_coords(ylabel_x, ylabel_y)
    ## gases (oxygen levels)
    # this is based on the experiment done in Albenberg et. al. (2014)
    # Figure 2, panel A, where they flowed in pure oxygen and measured
    # gut luminal oxygen levels. Their measurements were based on
    # Oxyphor G4 probe (described here: http://www.ncbi.nlm.nih.gov/pubmed/21961699)
    ax4 = plt.subplot(gs[3, 0])
    oxy_x = [0, 4, 5, 10]
    oxy_y = [60, 60, 40, 0.5]
    plt.step(oxy_x, oxy_y, color="k", clip_on=False)
    plt.yticks(range(0, 80, 20))
    plt.xlim(0 - c, nutr1_x[-1] + c)
    plt.xticks(time_axis)
    plt.xlabel("Time step", fontsize=8)
    plt.ylabel("p$O_{2}$ (mmHg)", fontsize=labels_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=8,
                    pad=2)
    ax4.yaxis.set_label_coords(ylabel_x, ylabel_y)
    # plot the schematic
    gs2 = gridspec.GridSpec(1, 1)
    gs2.update(left=0.15, right=0.8,
               top=0.23, bottom=0.12)
    ax = plt.subplot(gs2[0, 0])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axis("off")
    plot_schematic(fig, ax, panel_label_x=panel_label_x)
    sns.despine(trim=True, offset=1.)
    #plt.subplots_adjust(hspace=-0.2)
    # include legend when saving figure
    plt.savefig(plot_fname)
#                bbox_extra_artists=(lgd,))

def plot_schematic(fig, ax, panel_label_x=-0.17):
    coords_type = "figure fraction"
    panel_coords_type = "figure fraction"
    # The cell
    plt.plot([0.72], [0.1], 'o',
             markersize=65,
             markeredgecolor="k",
             markeredgewidth=1,
             markerfacecolor='None',
             clip_on=False)
    # "Cell"
    cell_x = 0.4
    cell_y = 0.2
    plt.annotate("Cell", xy = (cell_x, cell_y),
                 xycoords = coords_type,
                 xytext = (cell_x, cell_y),
                 textcoords = coords_type,
                 fontsize = 10,
                 fontstyle="italic",
                 color = "k")
    # annotate panel as B
    plt.annotate("B", xy=(panel_label_x, 0.25),
                 xycoords=panel_coords_type,
                 fontsize=PANEL_LABEL_FONTSIZE,
                 fontweight="bold")
    # "Environment"
    env_x = 0.1
    env_y = 0.12
    plt.annotate("Environment", xy = (env_x, env_y),
                 xycoords = coords_type,
                 xytext = (env_x, env_y),
                 textcoords = coords_type,
                 fontsize = 10,
                 fontstyle="italic",
                 color = "k")
    # arrow from Environment to Cell
    env_arrow_x = 0.375
    env_arrow_y = 0.13
    plt.annotate('', xy = (env_arrow_x, env_arrow_y),
                 xycoords = coords_type,
                 xytext = (env_arrow_x + 0.08, env_arrow_y),
                 textcoords = coords_type,
                 fontsize = 10,
                 color = "k",
                 arrowprops=dict(edgecolor='black',
                                 arrowstyle = '<|-',
                                 facecolor="k",
                                 linewidth=1,
                                 shrinkA = 0,
                                 shrinkB = 0))
    # arrow from Inference to Action
    plt.annotate('', xy = (0.62, 0.125),
                 xycoords = coords_type,
                 xytext = (0.62, 0.08),
                 textcoords = coords_type,
                 fontsize = 10, \
                 color = "k",
                 arrowprops=dict(edgecolor='black',
                                 arrowstyle = '<|-',
                                 facecolor="k",
                                 linewidth=1,
                                 shrinkA = 0,
                                 shrinkB = 0))
    # "Action"
    action_x = 0.555
    action_y = 0.05
    plt.annotate("Action", xy = (action_x, action_y),
                 xycoords = coords_type,
                 xytext = (action_x, action_y),
                 textcoords = coords_type,
                 fontsize = 10, 
                 color = "k")
    # "Inference"
    inf_x = 0.523
    inf_y = 0.145
    plt.annotate("Inference", xy = (inf_x, inf_y), fontsize=10,
                 xycoords=coords_type,
                 bbox={'facecolor': "#999999",
                       'alpha': 0.7,
                       'pad': 4})
    # plt.text(0.55, 0.1, "Inference", fontsize=10,
    #          bbox={'facecolor': "#999999",
    #                'alpha': 0.7,
    #                'pad': 4})


@follows(make_figure1)
@files(run_analyses,
       os.path.join(FIGS_DIR, "figure2.pdf"),
       "figure2")
def make_figure2(input_fname, plot_fname, label,
                 panel_label_x=0.5):
    """
    Figure 2: Markovian environments and when probabilistic structure
    of the environment affects fitness.
    """
    sns.set_style("ticks")
    width = 6
    height = 4.
    fig = plt.figure(figsize=(width, height))
    # plot panel A
    gs1 = gridspec.GridSpec(6, 3)
    gs1.update(left=0.02, bottom=0.1, right=0.98, wspace=0.1, hspace=1.8)
    plot_figure2_panel_A(fig, gs1)
    # plot panel B
    gs2 = gridspec.GridSpec(1, 3)
    gs2.update(left=0.082, top=0.6, right=0.88, wspace=0.5, hspace=0.01)
    plot_figure2_panel_B(fig, gs2)
    plt.savefig(plot_fname)
    
def plot_figure2_panel_A(main_ax, gs,
                         labels_fontsize=10,
                         panel_label_x=0.01):
    """
    Figure 2: Markovian environments
    """
    positions = [[0, 0],
                 [0, 1],
                 [0, 2],
                 [1, 0],
                 [1, 1],
                 [1, 2]]
    # set seed
    np.random.seed(2)
    # transition parameters: glu->glu, and gal->glu
    trans_params = [(0.9, 0.1),
                    (0.5, 0.5),
                    (0.1, 0.9),
                    (0.3, 0.3),
                    (0.9, 0.9),
                    (0.1, 0.1)]
    time_obj = time_unit.Time(0, 100, step_size=1)
    num_points = len(time_obj.t)
    # prior on carbon states
    p_carbon_state = 0.5
    num_plots = len(trans_params)
    n = 0
    data_to_labels = {True: "Glu",
                      False: "Gal"}
    labels_to_colors = \
      {"Glu": plot_utils.lightgreen,
       "Gal": plot_utils.blue}
    sns.set_style("ticks")
    num_xticks = 6
    for n in range(num_plots):
        ax = plt.subplot(gs[positions[n][0], positions[n][1]])
        if n == 0:
            ax.annotate("A", xy=(panel_label_x, 0.96),
                        xytext=(panel_label_x, 0.96),
                        xycoords="figure fraction",
                        textcoords="figure fraction",
                        fontsize=PANEL_LABEL_FONTSIZE,
                        fontweight="bold")
        p_gluc_to_gluc, p_galac_to_gluc = trans_params[n]
        samples = \
          model_sudden_markov.simulate_nutrient_trans(num_points,
                                                      p_gluc_to_gluc,
                                                      p_galac_to_gluc,
                                                      p_carbon_state=p_carbon_state)
        plot_utils.plot_sudden_switches(time_obj, samples,
                                        data_to_labels=data_to_labels,
                                        labels_to_colors=labels_to_colors,
                                        ax=ax)
        plt.xticks(np.linspace(time_obj.t_start, time_obj.t_end, num_xticks),
                   fontsize=8)
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        ax.spines["left"].set_visible(False)
        if n == 4:
            plt.xlabel("Time step", fontsize=labels_fontsize)
        plt.title(r"$\theta_{\mathsf{Glu}\rightarrow\mathsf{Glu}} = %.2f$, " \
                  r"$\theta_{\mathsf{Gal}\rightarrow\mathsf{Glu}} = %.2f$" \
                  %(p_gluc_to_gluc,
                    p_galac_to_gluc),
                  fontsize=8)
        
def plot_figure2_panel_B(main_ax, gs,
                         panel_label_x=0.01):
    params = simulation.load_params(pipeline_sudden.MARKOV_PARAM_FNAME)
    #params["gluc_growth_rate"] = 0.3
    #galac_growth_rates = [0.075, 0.15, 0.28]
    params["gluc_growth_rate"] = 0.7
    galac_growth_rates = [0.175, 0.35, 0.68]
    # 4-fold lower growth rate, 2-fold, nearly equal growth rates
    num_plots = len(galac_growth_rates)
    sns.set_style("white")
    for plot_num, galac_growth_rate in enumerate(galac_growth_rates):
        show_x_label = False
        show_y_label = False
        show_colorbar_label = False
        params["galac_growth_rate"] = galac_growth_rate
        ax = plt.subplot(gs[0, plot_num])
        if plot_num == 0:
            ax.annotate("B", xy=(panel_label_x, 0.51),
                        xytext=(panel_label_x, 0.51),
                        xycoords="figure fraction",
                        textcoords="figure fraction",
                        fontsize=PANEL_LABEL_FONTSIZE,
                        fontweight="bold")
        plt.gca().set_aspect("equal")
        if (plot_num + 1) == 1:
            show_x_label = False
            show_y_label = True
        elif (plot_num + 1) == 2:
            show_x_label = True
            show_y_label = False
        elif (plot_num + 1) == 3:
            show_x_label = False
            show_colorbar_label = True
        model_sudden_markov.plot_decision_mat(params,
                                              show_x_label=show_x_label,
                                              show_y_label=show_y_label,
                                              show_colorbar_label=show_colorbar_label)
        plt.tick_params(axis='both', which='major', labelsize=8,
                        pad=1)


@follows(make_figure2)
@files(pipeline_sudden.make_sudden_ssm_model_comparison,
       os.path.join(FIGS_DIR, "figure3.pdf"),
       "figure3")
def make_figure3(input_fname, plot_fname, label,
                 panel_label_x=0.1):
    """
    Figure 3: Simplified graphical model and example of model
    particle filter between periodic and constant environments.
    """
    # load model results
    all_results = simulation.load_data(input_fname)
    all_results = all_results["model"]
    # take only the second simulation of the switch SSM
    # model
    num_models = len(all_results)
    if num_models < 2:
        raise Exception, "Expected two switch SSM models."
    fig = plt.figure(figsize=(5.8, 3.5))
    gs = gridspec.GridSpec(2, 1,
                           height_ratios=[0.2, 0.05])
    gs.update(hspace=0.8, top=0.55, bottom=0.15)
    data_label = all_results.keys()[1]
    print "plotting switch ssm model %s" %(data_label)
    data_set = all_results[data_label]
    data_set = all_results[data_label]
    params = data_set["params"]
    time_obj = time_unit.Time(params["t_start"],
                              params["t_end"],
                              step_size=params["step_size"])
    c = 0.8
    x_axis = time_obj.t[0::4]
    xlims = [time_obj.t[0] - c, time_obj.t[-1] + c]
    ax1 = plt.subplot(gs[0, 0])
    pred_probs = [p[0] for p in data_set["preds_with_lag"]]
    panel_label_x = 0.01
    plt.annotate("A",
                 xy=(panel_label_x, 0.925),
                 xycoords="figure fraction",
                 fontweight="bold",
                 fontsize=PANEL_LABEL_FONTSIZE)
    plt.annotate("B",
                 xy=(panel_label_x, 0.61),
                 xycoords="figure fraction",
                 fontweight="bold",
                 fontsize=PANEL_LABEL_FONTSIZE)
    plt.plot(time_obj.t, pred_probs, "-o", color=plot_utils.red,
             label="Meta-change model",
             markersize=5,
             clip_on=False,
             zorder=100)
    plt.ylabel(r"$P(C_{t+1} =\ \mathsf{Glu} \mid  C_{0:t})$",
               fontsize=10)
    plt.xticks(x_axis, fontsize=8)
    plt.xlim(xlims)
    plt.ylim([0, 1])
    plt.yticks(np.arange(0, 1 + 0.2, 0.2),
               fontsize=8)
    ## annotate posterior inference trajectory
    # "Slow adaptation" 
    plt.annotate("Slow adaptation",
                 xy=(23, 0.61),
                 xytext=(18, 1.03),
                 xycoords="data",
                 textcoords="data",
                 fontsize=10,
                 arrowprops=dict(edgecolor='black',
                                 arrowstyle = '-|>',
                                 connectionstyle="angle,angleA=0,angleB=90",
                                 facecolor="k",
                                 linewidth=1,
                                 shrinkA = 0,
                                 shrinkB = 1))
    # "Faster adaptation" (2nd exposure)
    plt.annotate("Faster adaptation",
                 xy=(61.5, 0.63),
                 xytext=(57, 1.2),
                 xycoords="data",
                 textcoords="data",
                 fontsize=10,
                 arrowprops=dict(edgecolor='black',
                                 arrowstyle = '-|>',
                                 connectionstyle="angle,angleA=0,angleB=90",
                                 facecolor="k",
                                 linewidth=1,
                                 shrinkA = 0,
                                 shrinkB = 4))
    # as control model, fit "flat" Bayesian model
    data = params["data"]
    flat_bayes = \
      model_flat_bayes.FlatBayesModel(2, alpha_prior=np.array([1., 1.]))
    flat_probs = flat_bayes.predict(data)
    print flat_probs
    print flat_probs[:, 0]
    plt.plot(time_obj.t, flat_probs[:, 0], ":", color="k",
             label="Flat Markov model",
             linewidth=0.5,
             zorder=200,
             clip_on=False)
    # plot legend
    plt.legend(loc=(0.8, 0.1),
               fontsize=8)
    ##
    ## plot environment
    ##
    ax2 = plt.subplot(gs[1, 0])
    data_to_labels = {0: "Glu", 1: "Gal"}
    labels_to_colors = {"Glu": plot_utils.green, "Gal": plot_utils.blue}
    ax2.get_yaxis().set_visible(False)
    ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)
    plot_utils.plot_sudden_switches(time_obj, data,
                                    data_to_labels=data_to_labels,
                                    labels_to_colors=labels_to_colors,
                                    box_height=0.025,
                                    y_val=0.02,
                                    ax=ax2,
                                    despine=False,
                                    with_legend=True,
                                    legend_outside=(0,1))
    trans = ax2.get_xaxis_transform() # x in data untis, y in axes fraction
    ## annotate hidden switch states
    # periodic
    boxstyle = "round"
    linewidth = 0.8
    plt.annotate("Periodic", xy=(10, 1.5), xycoords=trans,
                 va="center", ha="center",
                 fontsize=8,
                 annotation_clip=False,
                 bbox=dict(boxstyle=boxstyle, fc="w",
                           linewidth=linewidth))
    # constant
    plt.annotate("Constant", xy=(30, 1.5), xycoords=trans,
                 va="center", ha="center",
                 fontsize=8,
                 annotation_clip=False,
                 bbox=dict(boxstyle=boxstyle, fc="w",
                           linewidth=linewidth))
    # periodic
    plt.annotate("Periodic", xy=(50, 1.5), xycoords=trans,
                 va="center", ha="center",
                 fontsize=8,
                 annotation_clip=False,
                 bbox=dict(boxstyle=boxstyle, fc="w",
                           linewidth=linewidth))
    # constant
    plt.annotate("Constant", xy=(70, 1.5), xycoords=trans,
                 va="center", ha="center",
                 fontsize=8,
                 annotation_clip=False,
                 bbox=dict(boxstyle=boxstyle, fc="w",
                           linewidth=linewidth))
    plt.xlabel("Time step", fontsize=10)
    plt.xticks(x_axis, fontsize=8)
    plt.xlim(xlims)
    # despine axes
    sns.despine(trim=True, left=True, ax=ax2)
    sns.despine(trim=True, ax=ax1)
    pad = 0.7
    ticklen = 3.0
    ax2.tick_params(axis="both", which="both", pad=pad,
                    length=ticklen,
                    labelsize=8)
    ax1.spines["left"].set_visible(True)
    ax1.tick_params(axis="both", which="both", pad=pad,
                    length=ticklen,
                    labelsize=8)
    plt.savefig(plot_fname)


def plot_fitness_simulation(df, params,
                            title=None,
                            yticks=None,
                            x_step=None,
                            ymax=None,
                            legend=False):
    sns.set_style("ticks")
   # plot population size
    popsizes = fitness.str_popsizes_to_array(df["log_pop_sizes"])
    ymin = popsizes.min()
    # take the total population size: sum of populations tuned to any
    # nutrient state
    df["log2_pop_size"] = np.log2(np.exp(popsizes).sum(axis=1))
    policy_colors = params["policy_colors"]
    # only plot policies we have colors for
    policies_with_colors = policy_colors.keys()
    df = df[df["policy"].isin(policies_with_colors)]
    init_pop_size = sum(params["init_pop_sizes"])
    # group results by time and by policy
    grouped = df.groupby(["t", "policy"], as_index=False)
    summary_df = grouped.agg({"log2_pop_size": [np.mean, np.std],
                              "growth_rates": [np.mean, np.std]})
    rand_summary = summary_df[summary_df["policy"] == "Random"]
    time_obj = time_unit.Time(params["t_start"],
                              params["t_end"],
                              step_size=params["step_size"])
    step_size = params["step_size"]
    final_df = summary_df[summary_df["t"] == time_obj.t[-1]]
    ## plot population size across time
    def plot_pop_size_across_time(params,
                                  ymin=ymin,
                                  ymax=ymax):
        offset = step_size / 250.
        num_xticks = 11
        ax = sns.tsplot(time="t", value="log2_pop_size", unit="sim_num",
                        condition="policy", color=policy_colors,
                        err_style="ci_band",
                        ci=95,
                        data=df,
                        legend=legend)
        for policy_num, policy in enumerate(policy_colors):
            error_df = summary_df[summary_df["policy"] == policy]
            c = policy_colors[policy]
            assert (len(error_df["t"]) == len(time_obj.t) == \
                    len(error_df["log2_pop_size"]["mean"])), \
              "Dataframe values for pop. size don\'t match time units."
        plt.xlabel("")
        plt.ylabel("")
#        plt.xlabel("Time step", fontsize=10)
#        plt.ylabel("Pop. size ($\log_{2}$)", fontsize=10)
        # assuming glucose is listed first
        gluc_growth_rate = params["nutr_growth_rates"][0]
        galac_growth_rate = params["nutr_growth_rates"][1]
        if title is not None:
            plt.title(title, fontsize=8)
        else:
            plt.title(r"$P_{0} = %d$, " \
                      r"$\mu_{\mathsf{Glu}} = %.2f, \mu_{\mathsf{Gal}} = %.2f$, " \
                      r"$\mu_{\mathsf{Mis}} = %.2f$, lag = %d, " \
                      r"%d iters" %(init_pop_size,
                                    gluc_growth_rate,
                                    galac_growth_rate,
                                    params["mismatch_growth_rate"],
                                    params["decision_lag_time"],
                                    params["num_sim_iters"]),
                     fontsize=8)
        c = 0.5
        plt.xlim([min(df["t"]) - c, max(df["t"]) + c])
        if ymin is None:
            ymin = int(np.log2(init_pop_size))
        plt.ylim(ymin=ymin)
        plt.xlim([time_obj.t.min(),
                  time_obj.t.max()])
        plt.xticks(range(int(time_obj.t.min()),
                         int(time_obj.t.max()) + x_step,
                             x_step),
                   fontsize=8)
        if yticks is not None:
            plt.yticks(yticks, fontsize=8)
            plt.ylim(yticks[0], yticks[-1])
        sns.despine(trim=True, offset=2*time_obj.step_size,
                    ax=ax)
        ax.tick_params(axis='both', which='major', labelsize=8,
                       pad=2)
        return ax
    # make plot
    ax = plot_pop_size_across_time(params)
    return ax


@follows(make_figure3)
@files(pipeline_sudden.merge_switch_ssm_fitness_sims,
       os.path.join(FIGS_DIR, "figure4.pdf"),
       "figure4")
def make_figure4(input_fname, plot_fname, label):
    """
    Figure 4: Population size simulations for different growth
    strategies.
    """
    print "plotting fitness simulation for switch ssm"
    sim_info = simulation.load_data(input_fname)
    results = sim_info["data"]
    # get the simulations that should be plotted based on their
    # parameter values
    params_to_plot = [{"p_switch_to_switch": 0.1,
                       "p_noswitch_to_switch": 0.1},
                      {"p_switch_to_switch": 0.1,
                       "p_noswitch_to_switch": 0.95},
                      {"p_switch_to_switch": 0.95,
                       "p_noswitch_to_switch": 0.1},
                      {"p_switch_to_switch": 0.95,
                       "p_noswitch_to_switch": 0.95}]
    sims_to_plot = []
    for sim_name in results:
        # see if current simulation matches the parameters
        # we're looking for
        for curr_params in params_to_plot:
            if len(results[sim_name]["params"]["nutr_labels"]) != 2:
                # skip any simulation that doesn't have two nutrients
                continue
            if utils.all_match(curr_params, results[sim_name]["params"]):
                sims_to_plot.append(sim_name)
    subplot_pos = [[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]]
    num_plots = len(sims_to_plot)
    fig = plt.figure(figsize=(3.3, 5))
    ax = fig.add_subplot(111)
    sns.set_style("ticks")
    y_step = 10
    max_y = 40
    yticks = np.arange(10, max_y + y_step, y_step)
    x_step = 20
    assert (num_plots == 4 == len(subplot_pos)), \
      "Expected 4 simulations to plot."
    gs = gridspec.GridSpec(num_plots / 2, num_plots)
    top = 0.70
    gs.update(right=1.75, left=0.15, top=top, hspace=0.45,
              wspace=0.4)
    axes = []
    for n, sim_to_plot in enumerate(sims_to_plot):
        if sim_to_plot not in results:
            raise Exception, "No simulation %s" %(sim_to_plot)
        params = results[sim_to_plot]["params"]
        print "PLOTTING PARAMS: ", params
        params["policy_colors"] = \
          {"Random": plot_utils.blue,
           "Plastic": plot_utils.green,
           "Posterior predictive": plot_utils.red,
           "Glucose-only": plot_utils.purple}
        #plt.subplot(2, int(round(num_plots / 2.)), n + 1)
        ax = plt.subplot(gs[subplot_pos[n][0], subplot_pos[n][1]])
        gluc_val = params["nutr_labels"].index("glucose")
        galac_val = params["nutr_labels"].index("galactose")
        gluc_growth_rate = params["nutr_growth_rates"][gluc_val]
        galac_growth_rate = params["nutr_growth_rates"][galac_val]
        title = r"$p_1 = %.2f, p_2 = %.2f$" \
                 %(params["p_switch_to_switch"],
                   params["p_noswitch_to_switch"])
        sim_results = results[sim_to_plot]["data"]
        legend = False
        if n == 0:
            legend = True
        ax = plot_fitness_simulation(sim_results, params,
                                     title=title,
                                     yticks=yticks,
                                     x_step=x_step,
                                     legend=legend)
        axes.append(ax)
    # plot legend for policies
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels,
                   bbox_to_anchor=(0.05, 1.51),
                   loc=2,
                   ncol=2,
#                   labelspacing=0.2,
                   borderaxespad=0,
                   fontsize=8)
    # set labels for subplots
    plt.annotate("Time",
                 xy=(0.46, 0.01),
                 xycoords="figure fraction",
                 fontsize=10)
    plt.annotate(r"Population size ($\log_{2}$)",
                 xy=(0.01, 0.53),
                 xycoords="figure fraction",
                 rotation=90,
                 fontsize=10)
    # panel A label
    plt.annotate("A",
                 xy=(0.01, 0.96),
                 xycoords="figure fraction",
                 fontweight="bold",
                 fontsize=PANEL_LABEL_FONTSIZE)
    # panel B label
    plt.annotate("B",
                 xy=(0.01, 0.81),
                 xycoords="figure fraction",
                 fontweight="bold",
                 fontsize=PANEL_LABEL_FONTSIZE)
    ##
    ## plot schematic illustrating what p_1 and p_2 are
    ##
    out_trans_mat1 = results[sims_to_plot[0]]["params"]["out_trans_mat1"]
    out_trans_mat2 = results[sims_to_plot[0]]["params"]["out_trans_mat2"]
    plot_transitions_schematic(results, out_trans_mat1, out_trans_mat2,
                               p1_x_offset=-0.5,
                               p2_minus_x_offset=0.1,
                               p1_minus_y_offset=0.0)
    plt.savefig(plot_fname)


@follows(make_figure4)
@files(pipeline_sudden.merge_switch_ssm_fitness_sims,
       os.path.join(FIGS_DIR, "figure5.pdf"),
       "figure5")
def make_figure5(input_fname, plot_fname, label):
    """
    Figure 5: fitness simulation for multiple nutrients.
    """
    print "plotting fitness simulation for switch ssm with multi-nutrients"
    ##
    ## plot schematic of multinutrient environments
    ##
    fig = plt.figure(figsize=(3.3, 7))
    multi_schematic_top = 0.98
    plot_multinutrient_schematic(left=0.02,
                                 top=multi_schematic_top,
                                 bottom=0.71,
                                 right=0.97,
                                 wspace=0.01,
                                 hspace=0.3)
    ###
    ### plot simulation results
    ###
    sim_info = simulation.load_data(input_fname)
    results = sim_info["data"]
    # make the plot here for three of the simulations
    params_to_plot = [{"p_switch_to_switch": 0.1,
                       "p_noswitch_to_switch": 0.1},
                      {"p_switch_to_switch": 0.1,
                       "p_noswitch_to_switch": 0.95},
                      {"p_switch_to_switch": 0.95,
                       "p_noswitch_to_switch": 0.1},
                      {"p_switch_to_switch": 0.95,
                       "p_noswitch_to_switch": 0.95}]
    sims_to_plot = []
    for sim_name in results:
        # see if current simulation matches the parameters
        # we're looking for
        for curr_params in params_to_plot:
            if len(results[sim_name]["params"]["nutr_labels"]) != 3:
                # skip any simulation that doesn't have three nutrients
                continue
            if utils.all_match(curr_params, results[sim_name]["params"]):
                sims_to_plot.append(sim_name)
    subplot_pos = [[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]]
    num_plots = len(sims_to_plot)
    y_step = 10
    max_y = 30
    yticks = np.arange(10, max_y + y_step, y_step)
    x_step = 20
    assert (num_plots == 4 == len(subplot_pos)), \
      "Expected 4 simulations to plot."
    gs = gridspec.GridSpec(num_plots / 2, num_plots)
    sim_top = 0.48
    gs.update(right=1.75, left=0.15,
              top=sim_top,
              bottom=0.0755,
              hspace=0.45, wspace=0.4)
    axes = []
    for n, sim_to_plot in enumerate(sims_to_plot):
        if sim_to_plot not in results:
            raise Exception, "No simulation %s" %(sim_to_plot)
        params = results[sim_to_plot]["params"]
        params["policy_colors"] = OrderedDict()
        params["policy_colors"]["Posterior predictive"] = plot_utils.red
        params["policy_colors"]["Random"] = plot_utils.blue
        params["policy_colors"]["Plastic"] = plot_utils.green
        params["policy_colors"]["Glucose-only"] = plot_utils.purple
        #plt.subplot(2, int(round(num_plots / 2.)), n + 1)
        ax = plt.subplot(gs[subplot_pos[n][0], subplot_pos[n][1]])
        gluc_val = params["nutr_labels"].index("glucose")
        galac_val = params["nutr_labels"].index("galactose")
        gluc_growth_rate = params["nutr_growth_rates"][gluc_val]
        galac_growth_rate = params["nutr_growth_rates"][galac_val]
        title = r"$p_1 = %.2f, p_2 = %.2f$" \
                 %(params["p_switch_to_switch"],
                   params["p_noswitch_to_switch"])
        sim_results = results[sim_to_plot]["data"]
        legend = False
        if n == 0:
            legend = True
        ax = plot_fitness_simulation(sim_results, params,
                                     title=title,
                                     yticks=yticks,
                                     x_step=x_step,
                                     legend=legend)
        axes.append(ax)
    # plot legend for policies
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels,
                   bbox_to_anchor=(0.05, 1.51),
                   loc=2,
                   ncol=2,
#                   labelspacing=0.2,
                   borderaxespad=0,
                   fontsize=8)
    # set labels for subplots
    plt.annotate("Time",
                 xy=(0.46, 0.01),
                 xycoords="figure fraction",
                 fontsize=10)
    plt.annotate(r"Population size ($\log_{2}$)",
                 xy=(0.01, sim_top - 0.125),
                 xycoords="figure fraction",
                 rotation=90,
                 fontsize=10)
    # panel A label
    plt.annotate("A",
                 xy=(0.01, multi_schematic_top - 0.01),
                 xycoords="figure fraction",
                 fontweight="bold",
                 fontsize=PANEL_LABEL_FONTSIZE)
    # panel B label
    plt.annotate("B",
                 xy=(0.01, sim_top + 0.15),
                 xycoords="figure fraction",
                 fontweight="bold",
                 fontsize=PANEL_LABEL_FONTSIZE)
    ##
    ## plot schematic illustrating what p_1 and p_2 are
    ##
    # pull output transition matrices from first simulation plotted
    out_trans_mat1 = results[sims_to_plot[0]]["params"]["out_trans_mat1"]
    out_trans_mat2 = results[sims_to_plot[0]]["params"]["out_trans_mat2"]
    #np.random.seed(20)
    np.random.seed(2)
    plot_transitions_schematic(results, out_trans_mat1, out_trans_mat2,
                               num_samples=50,
                               left=0.1,
                               right=0.93,
                               top=0.635,
                               wspace=0.2,
                               bottom=0.565,
                               p1_x_offset=-10.5,
                               p2_x_offset=-11.0,
                               p1_minus_x_offset=-0.95,
                               p1_minus_y_offset=-0.01,
                               p2_minus_x_offset=6.5)
    plt.savefig(plot_fname)
    

def plot_multinutrient_schematic(left=0.02,
                                 right=0.99,
                                 hspace=0.6,
                                 wspace=0.1,
                                 top=0.9,
                                 bottom=0.7):
    print "plotting multinutrient schematic"
    np.random.seed(2)
    # transition parameters
    trans_params = [np.array([[0.9, 0.05, 0.05],
                              [0.05, 0.9, 0.05],
                              [0.05, 0.05, 0.9]]),
                    np.array([[0.05, 0.9, 0.05],
                              [0.9, 0.05, 0.05],
                              [0.05, 0.05, 0.9]]),
                    np.array([[0.05, 0.9, 0.05],
                              [0.1, 0.6, 0.3],
                              [0.05, 0.05, 0.9]])]
    titles = ["persistent Glu, Gal, Mal",
              "periodic Glu/Gal, persistent Mal",
              "Glu predicts Gal, Gal predicts Mal, persistent Mal"]
    # initial state probabilities: start with Glu
    init_probs = np.array([1., 0., 0.])
    time_obj = time_unit.Time(0, 100, step_size=1)
    num_points = len(time_obj.t)
    num_plots = len(trans_params)
    n = 0
    data_to_labels = {0: "Glu",
                      1: "Gal",
                      2: "Mal"}
    labels_to_colors = \
      {"Glu": sns.color_palette("Set1")[2],
       "Gal": sns.color_palette("Set1")[1],
       "Mal": plot_utils.orange}
    sns.set_style("ticks")
    num_xticks = 11
    palette = "seismic"
    prev_n = 0
    prev_k = 0
    gs = gridspec.GridSpec(num_plots * 3, 3)
    gs.update(left=left, right=right,
              hspace=hspace, wspace=wspace,
              top=top, bottom=bottom)
    # order of nutrients along transition matrix
    nutr_order = ["Glu", "Gal", "Mal"]
    for n in xrange(num_plots):
        trans_mat = trans_params[n]
        trans_df = pandas.DataFrame(trans_mat)
        trans_df.columns = nutr_order
        trans_df.index = nutr_order
        ax1 = plt.subplot(gs[prev_n:prev_n+3, 0])
        heatmap_ax = sns.heatmap(trans_df, annot=True, cbar=False,
                                 cmap=plt.cm.gray_r,
                                 linewidth=0.1,
                                 linecolor="k",
                                 annot_kws={"fontsize": 6})
        heatmap_ax.set_aspect("equal")
        ax1.tick_params(axis="both", which="major", pad=0.01, length=2.5,
                        labelsize=8)
        # make y-labels of heatmap with no rotation
        plt.yticks(rotation=0)
        # rotate x-labels of heatmap
        plt.xticks(rotation=90)
        if n != (num_plots - 1):
            plt.axis("off")
        ax2 = plt.subplot(gs[prev_k+1:prev_k + 2, 1:3])
        prev_n += 3
        prev_k += 3
        # sample values from current transition matrix
        samples = prob_utils.sample_markov_chain(num_points,
                                                 init_probs,
                                                 trans_mat)
        handles_info = \
          plot_utils.plot_sudden_switches(time_obj, samples,
                                          data_to_labels=data_to_labels,
                                          labels_to_colors=labels_to_colors,
                                          ax=ax2,
                                          pad=0.01,
                                          despine=True)
        if n == (num_plots - 1):
            plt.xlabel("Time step", fontsize=10)
        plt.title(titles[n], fontsize=7)
    

def plot_transitions_schematic(results, out_trans_mat1, out_trans_mat2,
                               num_samples=10,
                               hspace=0.3,
                               wspace=0.01,
                               left=0.1,
                               right=0.88,
                               top=0.96,
                               bottom=0.87,
                               p1_x_offset=0.0,
                               p1_y_offset=0.0,
                               p1_minus_x_offset=0.0,
                               p1_minus_y_offset=-0.02,
                               p2_x_offset=0.0,
                               p2_y_offset=0.0,
                               p2_minus_x_offset=0.0,
                               p2_minus_y_offset=0.0,
                               box_width=None,
                               labelsize=8,
                               na_label="NA",
                               na_color="#999999"):
    y_val = 0.5
    box_height = 0.5
    with_legend = False
    despine = True
#    schematic_gs = gridspec.GridSpec(1, num_cols)
    schematic_gs = gridspec.GridSpec(2, 1)
    schematic_gs.update(bottom=bottom, top=top,
                        left=left, right=right,
                        wspace=wspace,
                        hspace=hspace)
    pad = 0.3
    tick_len = 2.5
    # initial probabilities concentrated on first state
    num_states = out_trans_mat1.shape[0]
    init_probs = np.zeros(num_states, dtype=np.float32)
    init_probs[0] = 1.
    # sample states for first output transition matrix
    samples1 = prob_utils.sample_markov_chain(num_samples,
                                              init_probs,
                                              out_trans_mat1)
    # sample states for second output transition matrix
    samples2 = prob_utils.sample_markov_chain(num_samples,
                                              init_probs,
                                              out_trans_mat2)
    # calculate how many items we want to plot
    steps_to_plot = num_samples * 2
    # now we need padding between
    t_pad = 5
    steps_to_plot += 5
    t_start = 1
    t_end = steps_to_plot
    t_step = 1
    first_t_start = t_start
    schematic_time = time_unit.Time(t_start, t_end, t_step)
    ax1 = plt.subplot(schematic_gs[0, 0])
    plt.xlim([t_start - schematic_time.step_size,
              t_end + schematic_time.step_size])
    coords_type = "data"
    top_bottom_diff = top - bottom
    right_left_diff = (right - left) + wspace
    # x of xy = 0.35
    # x of xytext = 0.21
    # arrow from first environment to itself
    y_offset = top - top_bottom_diff
    # y-offset for arrow from first environment
    # to itself
    first_env_yoffset = top_bottom_diff
    # x coordinate of first state 
    first_state_x = (first_t_start + len(samples1)) / 2
    first_state_y = y_val + 0.1
    ms = 6
    plt.plot([first_state_x],
             [first_state_y], "o",
             color="k",
             ms=ms,
             clip_on=False)
    lw = 1
    ##
    ## draw arrow back to state (self-transition)
    ##
    self_arrow_y = 0.1
    #plt.ylim([y_val, first_state_y + self_arrow_y + 2])
    plt.plot([first_state_x, first_state_x],
             [first_state_y, first_state_y + self_arrow_y],
             linewidth=lw,
             color="k",
             clip_on=False)
    # self-transition arrow (first state)
    ax1.annotate("", xy=(first_state_x, first_state_y + self_arrow_y),
                xytext=(first_state_x, first_state_y + 0.005), 
                xycoords=coords_type,
                textcoords=coords_type,
                transform=ax1.transAxes, 
                arrowprops=dict(arrowstyle="<|-",
                                connectionstyle="bar,fraction=-0.3,armB=-20,armA=-10",
                                ec="k",
                                facecolor="k",
                                linewidth=1,
                                shrinkA=3, shrinkB=0))
    # arrow from second environment to first
    second_arrow_x = 0.77 * right_left_diff
    second_arrow_len = 0.4
    right_xstart = right - schematic_time.t[-1] * 0.5
    # p_1 transition probability
    p1_ypos = y_offset + 3*top_bottom_diff
    p1_x = first_state_x - 3.2 + p1_x_offset
    p1_y = first_state_y + 0.03 + p1_y_offset
    ax1.annotate(r"$p_1$",
                 xy=(p1_x, p1_y),
                 xytext=(p1_x, p1_y),
                 xycoords=coords_type,
                 fontsize=10,
                 textcoords=coords_type)
    # draw second state
    second_state_x = num_samples + t_pad + (num_samples / 2.0)
    second_state_y = first_state_y
    plt.plot([second_state_x], [second_state_y], "o",
             ms=ms,
             color="k",
             clip_on=False)
    # self-transition arrow (second state)
    plt.plot([second_state_x, second_state_x],
             [second_state_y, second_state_y + self_arrow_y],
             linewidth=lw,
             color="k",
             clip_on=False)
    ax1.annotate("", xy=(second_state_x, second_state_y + self_arrow_y),
                xytext=(second_state_x, second_state_y + 0.005), 
                xycoords=coords_type,
                textcoords=coords_type,
                transform=ax1.transAxes,
                arrowprops=dict(arrowstyle="<|-",
                                connectionstyle="bar,fraction=0.3,armB=10,armA=10",
                                ec="k",
                                facecolor="k",
                                linewidth=1,
                                shrinkA=3, shrinkB=0))
    p2_minus_x = second_state_x + 2 + p2_minus_x_offset 
    p2_minus_y = second_state_y + 0.03 + p2_minus_y_offset
    ax1.annotate(r"$1 - p_2$",
                 xy=(p2_minus_x, p2_minus_y),
                 xytext=(p2_minus_x, p2_minus_y),
                 xycoords=coords_type,
                 fontsize=7.5,
                 color="#999999",
                 textcoords=coords_type)
    # draw transition from second to first state
    ax1.annotate("",
                 xy=(first_state_x, first_state_y),
                 xytext=(second_state_x, second_state_y),
                 xycoords=coords_type,
                 textcoords=coords_type,
                 color="k",
                 arrowprops=dict(edgecolor='black',
                                 connectionstyle="arc3,rad=0.0",
                                 arrowstyle = '-|>',
                                 facecolor="k",
                                 linewidth=1,
                                 shrinkA = 0,
                                 shrinkB = 3))
    # draw transition from first to second state
    ax1.annotate("",
                 xy=(first_state_x, first_state_y),
                 xytext=(second_state_x, second_state_y),
                 xycoords=coords_type,
                 textcoords=coords_type,
                 color="k",
                 arrowprops=dict(edgecolor='black',
                                 connectionstyle="angle3,angleA=-35,angleB=35",
                                 arrowstyle = '<|-',
                                 facecolor="k",
                                 linewidth=1,
                                 shrinkA = 3,
                                 shrinkB = 0))
    
    # p_2 transition probability
    p2_x = p1_x + num_samples + p2_x_offset
    p2_y = p1_y + p2_y_offset 
    ax1.annotate(r"$p_2$",
                 xy=(p2_x, p2_y),
                 xytext=(p2_x, p2_y),
                 xycoords=coords_type,
                 fontsize=10,
                 textcoords=coords_type)
    # 1 - p_1 probability
    p1_minus_x = p2_x - 0.5 + p1_minus_x_offset
    p1_minus_y = p2_y + 0.07 + p1_minus_y_offset
    ax1.annotate(r"$1 - p_1$",
                 xy=(p1_minus_x, p1_minus_y),
                 xytext=(p1_minus_x, p1_minus_y),
                 xycoords=coords_type,
                 fontsize=7.5,
                 color=plot_utils.darkgrey,
                 textcoords=coords_type)
    ax1.axis("off")
    ##
    ## plot nutrient tracks
    ##
    ax = plt.subplot(schematic_gs[1, 0])
    data_to_labels = {0: "Glu",
                      1: "Gal",
                      2: "Mal"}
    data = samples1
    labels_to_colors = \
      {"Glu": plot_utils.lightgreen,
       "Gal": plot_utils.blue,
       "Mal": plot_utils.orange}
    first_x_coords = \
      plot_utils.plot_seq(data, schematic_time, first_t_start,
               data_to_labels,
               labels_to_colors,
               box_height=box_height,
               y_val=y_val)
    plt.xlim([t_start - schematic_time.step_size,
              t_end + schematic_time.step_size])
#    plt.ylim([y_val, y_val + box_height])
#    plt.ylim([0, y_val + 4*box_height])
    if despine:
        sns.despine(trim=True, left=True, ax=ax)
        ax.get_yaxis().set_visible(False)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="both", which="both", pad=pad,
                       length=tick_len,
                       labelsize=labelsize)
    sns.despine(trim=True, ax=ax)
    # plot second sequence
    data = samples2
    # start next sequence after pad
    second_t_start = first_t_start + len(samples1) + t_pad
    plot_utils.plot_seq(data, schematic_time, second_t_start,
                        data_to_labels,
                        labels_to_colors,
                        box_height=box_height,
                        y_val=y_val)
    ax.axis("off")


@follows(make_figure5)
@files(None,
       os.path.join(FIGS_DIR, "figure6.pdf"),
       "figure6")
def make_figure6(input_fname, plot_fname, label):
    """
    Figure 6: sketch of molecular implementation of transition
    counter.
    """
    print "plotting molecular network figure"
    fig = plt.figure(figsize=(3.3, 6))
    gs = gridspec.GridSpec(3, 1)
    gs.update(hspace=0.5, left=0.19, top=0.55, bottom=0.08)
    panel_label_x = 0.01
    ax1 = plt.subplot(gs[0, 0])
    # plot panel label "A"
    plt.annotate("A",
                 xy=(panel_label_x, 0.96),
                 xycoords="figure fraction",
                 fontweight="bold",
                 fontsize=PANEL_LABEL_FONTSIZE)
    # plot panel label "B"
    plt.annotate("B",
                 xy=(panel_label_x, 0.55),
                 xycoords="figure fraction",
                 fontweight="bold",
                 fontsize=PANEL_LABEL_FONTSIZE)
    print "loading SBML model from %s" %(SBML_MODEL_FNAME)
    sbml_model = sbml.SBML(SBML_MODEL_FNAME)
    t_start = 0
    t_end = 250
    num_time_bins = 500
    times = np.linspace(t_start, t_end, num_time_bins)
    doser = sbml.DoseSched(t_start, t_end, num_time_bins)
    nutrient_val = 50
    # Glu
    doser.add_dose("Glu", 0, 49, nutrient_val)
    doser.add_dose("Glu", 49, 99, 0)
    # Gal
    doser.add_dose("Gal", 49, 99, nutrient_val)
    doser.add_dose("Gal", 99, 149, 0)
    # Glu
    doser.add_dose("Glu", 99, 149, nutrient_val)
    doser.add_dose("Glu", 149, 249, 0)
    # Gal
    doser.add_dose("Gal", 149, 199, nutrient_val)
    doser.add_dose("Glu", 199, 200, 0)
    results = sbml_model.simulate_with_doses(times, doser)
    sns.set_style("ticks")
    plt.tick_params(axis='both', which='major', labelsize=8,
                    pad=2)
    vars_to_plot = ["[Glu]", "[Gal]"]
    # offset for despine
    offset = 2
    # x-offset for nutrient plotting
    x_offsets = {"[Glu]": -1,
                 "[Gal]": 1}
    nutrients_to_colors = {"[Glu]": plot_utils.lightgreen,
                           "[Gal]": plot_utils.blue}
    # linewidth
    lw = 1.5
    for c in vars_to_plot:
        if c != "time":
            plt.plot(results["time"] + x_offsets[c], results[c],
                     label=c,
                     color=nutrients_to_colors[c],
                     linewidth=lw,
                     clip_on=True)
    # time axis
    x_start = t_start
    x_end = t_end
    x_step = 50
    labelspacing = 0.25
    handletextpad = 0.5
    plt.xlim([x_start, x_end])
    plt.xticks(range(x_start, x_end + x_step, x_step))
    ylims = plt.gca().get_ylim()
    plt.ylim([0 - 1, 52])
    plt.ylabel("Inputs", fontsize=8)
    ylabel_x = -0.14
    ylabel_y = 0.5
    ax1.yaxis.set_label_coords(ylabel_x, ylabel_y)
    ax1.legend(fontsize=8, handlelength=2.2, handletextpad=handletextpad,
               numpoints=4,
               labelspacing=labelspacing,
               bbox_to_anchor=(1.12, 1.0))
    sns.despine(trim=True, offset=offset)
    ax2 = plt.subplot(gs[1, 0])
    clip_on = False
    plt.tick_params(axis='both', which='major', labelsize=8,
                    pad=2)
    plt.plot(results["time"], results["[Glu_Sensor]"],
             label="[Glu Sensor]",
             linewidth=lw,
             clip_on=clip_on,
             color=plot_utils.lightgreen)
    plt.plot(results["time"], results["[Gal_Sensor]"],
             label="[Gal Sensor]",
             linewidth=lw,
             clip_on=clip_on,
             color=plot_utils.blue)
    ax2.yaxis.set_label_coords(ylabel_x, ylabel_y)
    plt.xlim([x_start, x_end])
    plt.xticks(range(x_start, x_end + x_step, x_step))
    plt.ylim([-0.5, 14])
    plt.yticks(range(0, 14 + 2, 4))
    plt.ylabel("Sensors", fontsize=8)
    ax2.legend(fontsize=8, handlelength=2.2, handletextpad=handletextpad,
               numpoints=4,
               labelspacing=labelspacing,
               bbox_to_anchor=(1.12, 1.1))
    sns.despine(trim=True, offset=offset)
    ax3 = plt.subplot(gs[2, 0])
    # plt.plot(results["time"], results["[Glu_Activator]"],
    #          label="[Glu Activator]",
    #          linewidth=lw,
    #          color="k",
    #          clip_on=clip_on)
    # plt.plot(results["time"], results["[Gal_Activator]"],
    #          label="[Gal Activator]",
    #          linestyle="--",
    #          linewidth=lw,
    #          clip_on=clip_on,
    #          color="k")
    plt.plot(results["time"], results["[Glu_to_Gal]"],
             label="[Glu-to-Gal]",
             linewidth=lw,
             clip_on=clip_on,
             color="k")
    line, = plt.plot(results["time"], results["[Gal_to_Glu]"],
             label="[Gal-to-Glu]",
             linewidth=lw,
             clip_on=clip_on,
             linestyle="--",
             color="k")
    line.set_dashes((3, 2))
    ax3.legend(fontsize=8, handlelength=2.2, handletextpad=handletextpad,
               numpoints=4,
               labelspacing=labelspacing,
               bbox_to_anchor=(0.45, 1.1))
    plt.tick_params(axis='both', which='major', labelsize=8,
                    pad=2)
    plt.xlim([x_start, x_end])
    plt.xticks(range(x_start, x_end + x_step, x_step))
    plt.ylim([0, 150])
    plt.yticks(range(0, 150 + 50, 50))
    ax3.yaxis.set_label_coords(ylabel_x, ylabel_y)
    sns.despine(trim=True, offset=offset)
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Transition counters", fontsize=8)
    plt.savefig(plot_fname)


##
## Supplementary figures
##
@follows(make_figure1)
@files(None,
       os.path.join(FIGS_DIR, "supp_figure1.pdf"),
       "supp_figure1")
def make_supp_figure1(input_fname, plot_fname, label):
    """
    Supp. Figure 1: growth rates on different carbon sources.
    """
    # Load Liti-Fay growth rates
    mean_growth_rates_fname = \
      os.path.join(paths.LITI_FAY_DIR, "liti_fay_growthrates_mean.txt")
    mean_growth_rates = pandas.read_table(mean_growth_rates_fname,
                                          sep="\t")
    print "mean growth rates: "
    print mean_growth_rates.head(n=10)
    plt.figure(figsize=(5., 4.5))
    gs = gridspec.GridSpec(2, 1)
    ax1 = plt.subplot(gs[0, 0])
    growth_rates_df = pandas.melt(mean_growth_rates, id_vars=["Strain"])
    strains = growth_rates_df["Strain"].unique()
    num_strains = len(strains)
#    carbon_sources = ["Glucose", "Fructose", "Sucrose",
#                      "Galactose", "Maltose", "Raffinose",
#                      "Ethanol", "Acetate (KAc)"]
    # plot only sugars
    carbon_sources = ["Glucose", "Fructose", "Sucrose",
                      "Galactose", "Maltose", "Raffinose"]
    # only plot selected carbon sources
    growth_rates_df = \
      growth_rates_df[growth_rates_df["variable"].isin(carbon_sources)]
    growth_rates_df.rename(columns={"variable": "Sugar"}, inplace=True)
    sns.violinplot(x="Sugar", y="value", 
                   color=plot_utils.darkgrey,
                   data=growth_rates_df,
                   ax=ax1)
#    handles, labels = ax.get_legend_handles_labels()
#    ax.legend(handles, labels,
#              bbox_to_anchor=(0.10, 1.25),
#              loc=2,
#              ncol=2,
#              borderaxespad=0,
#              fontsize=8)
    panel_label_x = 0.01
    plt.annotate("A",
                 xy=(panel_label_x, 0.96),
                 xycoords="figure fraction",
                 fontweight="bold",
                 fontsize=PANEL_LABEL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=8,
                    pad=2)
    plt.title("Growth rates on different sugars (%d yeast strains)" %(num_strains),
              fontsize=10)
    plt.ylabel("Doublings per hr", fontsize=10) 
    plt.ylim([0, 1.2])
    step = 0.2
    yticks = np.arange(0, 1.2 + step, step)
    plt.yticks(yticks)
#    plt.xlabel("Sugars", fontsize=10)
    plt.xlabel("")
    sns.despine(trim=True, offset=2.)
    ax2 = plt.subplot(gs[1, 0])
    # plot only glucose and galactose growth rates
    glu_gal_df = \
      growth_rates_df[growth_rates_df["Sugar"].isin(["Glucose", "Galactose"])]
    # compute mean growth rates for Glucose and Galactose
    # across all strains and display the ratio
    mean_glu_growth_rate = mean_growth_rates["Glucose"].mean()
    mean_galac_growth_rate = mean_growth_rates["Galactose"].mean()
    print "mean glucose growth rate across strains: %.2f" \
          %(mean_glu_growth_rate)
    print "mean galactose growth rate across strains: %.2f" \
          %(mean_galac_growth_rate)
    glu_galac_rate_ratio = mean_glu_growth_rate / mean_galac_growth_rate
    print "ratio of glu/gal mean growth rates: %.2f" \
          %(glu_galac_rate_ratio)
    all_glu_galac_ratios = \
      mean_growth_rates["Glucose"] / mean_growth_rates["Galactose"]
    sns.distplot(all_glu_galac_ratios, rug=True, hist=False,
                 color="k",
                 ax=ax2)
    plt.xlabel("Glucose / Galactose doublings per hr", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    c = 0.01
    plt.ylim([0 - c, 2.5])
    plt.tick_params(axis='both', which='major', labelsize=8,
                    pad=2)
    plt.annotate("B",
                 xy=(panel_label_x, 0.46),
                 xycoords="figure fraction",
                 fontweight="bold",
                 fontsize=PANEL_LABEL_FONTSIZE)
    sns.despine(trim=True, offset=2.)
    gs.update(left=0.15, right=0.95, hspace=0.4,
              top=0.91,
              bottom=0.12)
    plt.savefig(plot_fname)


@follows(make_supp_figure1)
@files(None,
       os.path.join(FIGS_DIR, "supp_figure2.pdf"),
       "supp_figure2")
def make_supp_figure2(input_fname, plot_fname, label):
    """
    Supp. Figure 2 is made in TeX.
    """
    plt.figure()
    plt.text(0.5, 0.5, "This figure will be made in TeX.")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(plot_fname)


@follows(make_supp_figure2)
@files(pipeline_sudden.merge_switch_ssm_fitness_sims,
       os.path.join(FIGS_DIR, "supp_figure3.pdf"),
       "supp_figure3")
def make_supp_figure3(input_fname, plot_fname, label):
    """
    Supp. Figure 3: bet-hedging fitness simulation.
    """
    print "plotting fitness simulation for switch ssm"
    sim_info = simulation.load_data(input_fname)
    results = sim_info["data"]
    # make the plot here for three of the simulations
    params_to_plot = [{"p_switch_to_switch": 0.1,
                       "p_noswitch_to_switch": 0.1},
                      {"p_switch_to_switch": 0.1,
                       "p_noswitch_to_switch": 0.95},
                      {"p_switch_to_switch": 0.95,
                       "p_noswitch_to_switch": 0.1},
                      {"p_switch_to_switch": 0.95,
                       "p_noswitch_to_switch": 0.95}]
    sims_to_plot = []
    for sim_name in results:
        # see if current simulation matches the parameters
        # we're looking for
        for curr_params in params_to_plot:
            if len(results[sim_name]["params"]["nutr_labels"]) != 2:
                # skip any simulation that doesn't have two nutrients
                continue
            if utils.all_match(curr_params, results[sim_name]["params"]):
                sims_to_plot.append(sim_name)
    subplot_pos = [[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]]
    num_plots = len(sims_to_plot)
    fig = plt.figure(figsize=(3.3, 5))
    ax = fig.add_subplot(111)
    sns.set_style("ticks")
    y_step = 10
    max_y = 40
    yticks = np.arange(10, max_y + y_step, y_step)
    x_step = 20
    assert (num_plots == 4 == len(subplot_pos)), \
      "Expected 4 simulations to plot."
    gs = gridspec.GridSpec(num_plots / 2, num_plots)
    sim_top = 0.70
    gs.update(right=1.75, left=0.15, top=sim_top, hspace=0.45,
              wspace=0.4)
    axes = []
    for n, sim_to_plot in enumerate(sims_to_plot):
        if sim_to_plot not in results:
            raise Exception, "No simulation %s" %(sim_to_plot)
        params = results[sim_to_plot]["params"]
        params["policy_colors"] = OrderedDict()
        params["policy_colors"]["Posterior pred. (BH)"] = plot_utils.red
        params["policy_colors"]["Random (BH)"] = plot_utils.blue
        params["policy_colors"]["Plastic"] = plot_utils.green
        #plt.subplot(2, int(round(num_plots / 2.)), n + 1)
        ax = plt.subplot(gs[subplot_pos[n][0], subplot_pos[n][1]])
        gluc_val = params["nutr_labels"].index("glucose")
        galac_val = params["nutr_labels"].index("galactose")
        gluc_growth_rate = params["nutr_growth_rates"][gluc_val]
        galac_growth_rate = params["nutr_growth_rates"][galac_val]
        # title = r"$\mu_{\mathsf{Glu}} = %.2f, \mu_{\mathsf{Gal}} = %.3f$, " \
        #         r"$\mu_{\mathsf{Mis}} = %.1f$, " \
        #         r"$p_1 = %.2f, p_2 = %.2f$" \
        #         %(gluc_growth_rate,
        #           galac_growth_rate,
        #           params["mismatch_growth_rate"],
        #           params["p_switch_to_switch"],
        #           params["p_noswitch_to_switch"])
        title = r"$p_1 = %.2f, p_2 = %.2f$" \
                 %(params["p_switch_to_switch"],
                   params["p_noswitch_to_switch"])
        sim_results = results[sim_to_plot]["data"]
        legend = False
        if n == 0:
            legend = True
        ax = plot_fitness_simulation(sim_results, params,
                                     title=title,
                                     yticks=yticks,
                                     x_step=x_step,
                                     legend=legend)
        axes.append(ax)
    # plot legend for policies
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels,
                   bbox_to_anchor=(0.05, 1.51),
                   loc=2,
                   ncol=2,
#                   labelspacing=0.2,
                   borderaxespad=0,
                   fontsize=8)
    # set labels for subplots
    plt.annotate("Time",
                 xy=(0.46, 0.01),
                 xycoords="figure fraction",
                 fontsize=10)
    plt.annotate(r"Population size ($\log_{2}$)",
                 xy=(0.01, 0.74 * sim_top),
                 xycoords="figure fraction",
                 rotation=90,
                 fontsize=10)
    # panel A label
    plt.annotate("A",
                 xy=(0.01, 0.96),
                 xycoords="figure fraction",
                 fontweight="bold",
                 fontsize=PANEL_LABEL_FONTSIZE)
    # panel B label
    plt.annotate("B",
                 xy=(0.01, 0.81),
                 xycoords="figure fraction",
                 fontweight="bold",
                 fontsize=PANEL_LABEL_FONTSIZE)
    ##
    ## plot schematic illustrating what p_1 and p_2 are
    ##
    # pull output transition matrices from first simulation plotted
    out_trans_mat1 = results[sims_to_plot[0]]["params"]["out_trans_mat1"]
    out_trans_mat2 = results[sims_to_plot[0]]["params"]["out_trans_mat2"]
    plot_transitions_schematic(results, out_trans_mat1, out_trans_mat2,
                               p1_x_offset=-0.5,
                               p2_minus_x_offset=0.1,
                               p1_minus_y_offset=0.0)
    plt.savefig(plot_fname)
    
def main():
    print "making meta-changing environments paper..."
    t1 = time.time()
    pipeline_run(multiprocess=4)
    t2 = time.time()
    print "analyses for paper took %.2f mins." %((t2 - t1) / 60.)

if __name__ == "__main__":
    main()
