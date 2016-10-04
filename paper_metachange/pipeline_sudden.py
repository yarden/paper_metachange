##
## Pipeline for sudden-switch experiments
##
import os
import sys
import time
import cPickle as pickle
from collections import OrderedDict

import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
sns.set_style("ticks")

import pandas

import numpy as np

import particlefever
import particlefever.switch_ssm as switch_ssm
import particlefever.particle_filter as particle_filter
import particlefever.sampler as sampler

import paths
import time_unit
import simulation
import fitness
import policies
import env
import nutrient_simulators
import model_sudden_markov
import model_switch_ssm
import plot_utils
import prob_utils
import utils

from ruffus import *

##
## default parameter filenames
##
# for Markov model
MARKOV_PARAM_FNAME = os.path.join(paths.SIM_PARAMS_DIR, "sudden_markov.params")
# for HMM model
HMM_PARAM_FNAME = os.path.join(paths.SIM_PARAMS_DIR, "sudden_hmm.params")
# for SSM model
SSM_2_NUTR_PARAM_FNAME = os.path.join(paths.SIM_PARAMS_DIR, "sudden_switch_ssm_2_nutr.params")
SSM_3_NUTR_PARAM_FNAME = os.path.join(paths.SIM_PARAMS_DIR, "sudden_switch_ssm_3_nutr.params")
## for bet-hedging simulation
# parameters for bet-hedging simulation with two nutrients
FITNESS_2_NUTR_SIM_PARAM_FNAME = \
  os.path.join(paths.SIM_PARAMS_DIR, "fitness_2_nutr_sim.params")
FITNESS_3_NUTR_SIM_PARAM_FNAME = \
  os.path.join(paths.SIM_PARAMS_DIR, "fitness_3_nutr_sim.params")
# parameters for bet-hedging simulation with three nutrients
BH_SIM_PARAMS_FNAMES = \
  {"2_nutr": FITNESS_2_NUTR_SIM_PARAM_FNAME,
   "3_nutr": None}

MARKOV_MODEL_NAME = "sudden_markov"
HMM_MODEL_NAME = "sudden_hmm"
SSM_MODEL_NAME = "sudden_ssm"

PLOTS_DIR = os.path.join(paths.RUFFUS_PLOTS_DIR, "sudden_switch")
utils.make_dir(PLOTS_DIR)

@originate(paths.PIPELINE_SUDDEN_START_FNAME)
def start_pipeline(output_fname):
    """
    Dummy function that starts pipeline.
    """
    with open(output_fname, "w") as out_file:
        out_file.write("starting pipeline sudden.\n")

##
## Markov tasks
##
@follows(start_pipeline)
@files(None,
       os.path.join(PLOTS_DIR, "sudden_markov_fixed_decision_mat.pdf"),
       "fixed_decision_mat")
def plot_sudden_markov_fixed_decision_mat(input_fname, plot_fname, label):
    """
    Plot decision matrix for sudden switch Markov model, with
    FIXED growth rates for glucose and galactose.
    """
    print "plotting: %s" %(os.path.basename(plot_fname))
    plt.figure(figsize=(5, 4))
    params = simulation.load_params(MARKOV_PARAM_FNAME)
#    params["gluc_growth_rate"] = 0.3
#    params["galac_growth_rate"] = 0.075
    params["gluc_growth_rate"] = 0.7
    params["galac_growth_rate"] = 0.175
    model_sudden_markov.plot_decision_mat(params)
    plt.tight_layout()
    plt.savefig(plot_fname)


@follows(plot_sudden_markov_fixed_decision_mat)
@files(None,
       os.path.join(PLOTS_DIR, "sudden_markov_vary_decision_mat.pdf"),
       "vary_decision_mat")
def plot_sudden_markov_variable_decision_mat(input_fname, plot_fname, label):
    """
    Plot decision matrix for sudden switch Markov model, with
    VARIABLE growth rates for glucose and galactose.
    """
    print "plotting: %s" %(os.path.basename(plot_fname))
    fig = plt.figure(figsize=(6, 3.1))
    params = simulation.load_params(MARKOV_PARAM_FNAME)
#    params["gluc_growth_rate"] = 0.3
#    galac_growth_rates = [0.075, 0.15, 0.28]
    params["gluc_growth_rate"] = 0.7
    # 4-fold lower growth rate, 2-fold, nearly equal growth rates
    galac_growth_rates = [0.175, 0.35, 0.68]
    num_plots = len(galac_growth_rates)
    # set heatmap aspect equal
    plt.gca().set_aspect("equal")
    sns.set_style("white")
    for plot_num, galac_growth_rate in enumerate(galac_growth_rates):
        show_x_label = False
        show_y_label = False
        show_colorbar_label = False
        params["galac_growth_rate"] = galac_growth_rate
        plt.subplot(1, num_plots, plot_num + 1)
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
    plt.subplots_adjust(wspace=0.5, left=0.1)
    fig.set_tight_layout(True)
    plt.savefig(plot_fname)

@follows(plot_sudden_markov_variable_decision_mat)
@files(SSM_2_NUTR_PARAM_FNAME,
       os.path.join(paths.SUDDEN_DATA_DIR, "sudden_ssm_comparison.model"),
       "sudden_switch_model_comparison")
def make_sudden_ssm_model_comparison(input_fname, output_fname, label):
    """
    Run discrete switching SSM model and compare the results for
    prediction, filtering and smoothing.
    """
    print "making switching SSM model from: %s" %(input_fname)
    data_sets = OrderedDict()
#    data_sets["period_noperiod"] = \
#      np.array([0, 1] * 10 + [0, 0] * 10)
    data_sets["period_noperiod_period"] = \
      np.array([0, 1] * 10 + [0, 0] * 10 + \
               [0, 1] * 10)
    data_sets["period_noperiod_period_noperiod"] = \
      np.array([0, 1] * 10 + [0, 0] * 10 + \
               [0, 1] * 10 + \
               [0, 0] * 10)
    all_results = OrderedDict()
    for data_label in data_sets:
        params = simulation.load_params(input_fname)
        init_switch_hyperparams = np.array(params["init_switch_hyperparams"])
        init_out_hyperparams = np.array(params["init_out_hyperparams"])
        switch_trans_mat_hyperparams = \
          np.array(params["switch_trans_mat_hyperparams"])
        out_trans_mat_hyperparams = np.array(params["out_trans_mat_hyperparams"])
        sticky_switch_weight = params["sticky_switch_weight"]
        num_switch_states = params["num_switch_states"]
        num_outputs = params["num_outputs"]
        num_particles = 200
        ssm_pf = \
          particle_filter.DiscreteSwitchSSM_PF(num_switch_states,
                                               num_outputs,
                                               num_particles=num_particles,
                                               init_switch_hyperparams=init_switch_hyperparams,
                                               init_out_hyperparams=init_out_hyperparams,
                                               switch_trans_mat_hyperparams=switch_trans_mat_hyperparams,
                                               out_trans_mat_hyperparams=out_trans_mat_hyperparams,
                                               sticky_switch_weight=sticky_switch_weight)
        data = data_sets[data_label]
        # time for simulation
        params["t_step"] = 1
        params["t_start"] = 0
        params["t_end"] = len(data) - params["t_step"]
        params["data"] = data
        # get filtering predictions
        print "running filtering predictions: "
        t1 = time.time()
        ssm_pf.initialize()
        ssm_pf.process_data(data, save_steps=True)
        t2 = time.time()
        print "filtering took %.2f" %(t2 - t1)
        # get predictions with lags
        print "getting predictions with lag"
        prev_output = data[-1]
        num_preds = 10
        t1 = time.time()
        preds_with_lag = ssm_pf.prediction_with_lag(data, lag=1)
        t2 = time.time()
        print "predictions with lag took %.2f seconds" %(t2 - t1)
        model_results = {"params": params,
                         "preds_with_lag": preds_with_lag,
                         "model": ssm_pf.prior}
        all_results[data_label] = model_results
    utils.save_model_helper(output_fname, all_results)

@files(make_sudden_ssm_model_comparison, 
       os.path.join(PLOTS_DIR, "sudden_ssm_pred_example.pdf"),
       "ssm_pred_example")
def plot_sudden_ssm_filtering(input_fname, plot_fname, label):
    """
    Plot SSM filtering predictions.
    """
    print "plotting: %s" %(os.path.basename(plot_fname))
    fig = plt.figure(figsize=(7, 5))
    sns.set_style("ticks")
    all_results = simulation.load_data(input_fname)
    all_results = all_results["model"]
    num_plots = len(all_results)
    total_plots = num_plots * 2
    gs = gridspec.GridSpec(total_plots, 1,
                           height_ratios=[1, 0.2]*num_plots)
    curr_plot_num = 0
    axes = {}
    for n, data_label in enumerate(all_results.keys()):
        data_set = all_results[data_label]
        params = data_set["params"]
        time_obj = time_unit.Time(params["t_start"],
                                  params["t_end"],
                                  step_size=params["step_size"])
        c = 0.8
        x_axis = time_obj.t[0::4]
        xlims = [time_obj.t[0] - c, time_obj.t[-1] + c]
        ax1 = plt.subplot(gs[curr_plot_num, 0])
        pred_probs = [p[0] for p in data_set["preds_with_lag"]]
        plt.plot(time_obj.t, pred_probs, "-o", color=plot_utils.red,
                 label="Prediction",
                 clip_on=False,
                 zorder=100)
        plt.xlabel(r"Time step")
        plt.ylabel(r"$P(C_{t+1} =\ \mathsf{Glu} \mid  C_{0:t})$",
                   fontsize=11)
        plt.title("lag = %d" %(params["decision_lag_time"]), fontsize=8)
        plt.legend(loc="lower right")
        plt.xticks(x_axis, fontsize=8)
        plt.xlim(xlims)
        plt.ylim([0, 1])
        plt.yticks(np.arange(0, 1 + 0.2, 0.2))
        ax2 = plt.subplot(gs[curr_plot_num + 1, 0])
        data_to_labels = {0: "Glu", 1: "Gal"}
        labels_to_colors = {"Glu": plot_utils.green, "Gal": plot_utils.blue}
        data = params["data"]
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
        plt.xticks(x_axis, fontsize=8)
        plt.xlim(xlims)
        # despine axes
        sns.despine(trim=True, left=True, ax=ax2)
        sns.despine(trim=True, ax=ax1)
        # advance number of plots by two
        curr_plot_num += 2
    ax1.spines["left"].set_visible(True)
    #plt.tight_layout(h_pad=0.1)
    fig.set_tight_layout(True)
    plt.savefig(plot_fname)

##
## Fitness simulations for SSM
## these depend on:
## 1. the fitness simulators parameters file, and
## 2. the model (SSM) parameters file
##
###
### TODO: write a get_parameters function here
##        that takes as first argument a list:
##           [model_param_fname, fitness_param_fname]
## 
##        and as remaining arguments what is needed
##        for the simulation.
##

###
### TODO: change this to handle multiple nutrients as another parameter
###                     
def get_switch_ssm_fitness_params():
    # parameters common to all nutrient simulations
    # (regardless of whether it's with 2 or 3 nutrients)
    all_p_switch_to_switch = [0.1, 0.95]
    all_p_noswitch_to_switch = [0.1, 0.95]
    all_p_init_switch = [0.5]
    ## 3-nutrient related variables
    # diagonal heavy: 3 sticky nutrients
    sticky_3_nutr_trans_mat = \
      np.array([[0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9]])
    # periodic glu/gal and sticky mal
    periodic_sticky_mal_trans_mat = \
      np.array([[0.05, 0.9, 0.05],
                [0.9, 0.05, 0.05],
                [0.05, 0.05, 0.9]])
    # glu predictive of gal; gal predictive of mal;
    # sticky mal state
    glu_pred_gal_pred_mal_trans_mat = \
      np.array([[0.05, 0.9, 0.05],
                [0.1, 0.6, 0.3],
                [0.05, 0.05, 0.9]])
    # 2-nutrient parameters
    nutrient_params = \
      {"2_nutr": \
       {"fitness_params_fname": FITNESS_2_NUTR_SIM_PARAM_FNAME,
        "model_params_fname": SSM_2_NUTR_PARAM_FNAME,
        "nutr_labels": ["glucose", "galactose"],
        # list of all growth rates to consider
        # format is [[gluc_growth_rate1, galac_growth_rate1],
        #            [gluc_growth_rate2, galac_growth_rate2],
        #            ...]
        "all_nutr_growth_rates": \
        [[0.3, 0.3/2.]],
        "all_out_trans_mats": \
        [[np.array([[0., 1],
                    [1., 0]]),
          np.array([[1., 0.],
                    [1., 0.]])]]},
       "3_nutr": \
       {"fitness_params_fname": FITNESS_3_NUTR_SIM_PARAM_FNAME,
        "model_params_fname": SSM_3_NUTR_PARAM_FNAME,
        "nutr_labels": ["glucose", "galactose", "maltose"],
        "all_nutr_growth_rates": \
        [[0.3, 0.3/2., 0.3/2.]],
        "all_out_trans_mats": \
        [[glu_pred_gal_pred_mal_trans_mat,
          periodic_sticky_mal_trans_mat]]}}
    # 3-nutrient parameters
    ### TODO: fill this in -- add matrices for 3 nutrient
    ### simulations
    fitness_params = {}
    sim_num = 1
    ### TODO: here, add 2-nutrient or 3-nutrient as a paramter
    ### so that the subsequent pipeline function can run on
    ### both 2-nutrient and 3-nutrient fitness simulations
    for sim_type in nutrient_params:
        curr_params = nutrient_params[sim_type]
        input_fnames = [curr_params["fitness_params_fname"],
                        curr_params["model_params_fname"]]
        for p_switch_to_switch in all_p_switch_to_switch:
            for p_noswitch_to_switch in all_p_noswitch_to_switch:
                for nutr_growth_rates in curr_params["all_nutr_growth_rates"]:
                    for out_trans_mats in curr_params["all_out_trans_mats"]:
                        sim_params = \
                          {"nutr_labels": curr_params["nutr_labels"],
                           "out_trans_mat1": out_trans_mats[0],
                           "out_trans_mat2": out_trans_mats[1],
                           "nutr_growth_rates": nutr_growth_rates,
                           "p_switch_to_switch": p_switch_to_switch,
                           "p_noswitch_to_switch": p_noswitch_to_switch,
                           "out_trans_mats": out_trans_mats}
                        sim_cond = "sim_%d" %(sim_num)
                        output_fname = \
                          os.path.join(paths.SUDDEN_DATA_DIR,
                                       "switch_ssm_fitness_sim%d.data" %(sim_num))
                        yield (input_fnames, output_fname, sim_params, sim_cond)
                        sim_num += 1


@files(get_switch_ssm_fitness_params)
def run_switch_ssm_fitness_simulations(input_fnames, output_fname,
                                       sim_params, sim_label):
    """
    Run switch SSM fitness simulations.
    Compare the results of different growth policies.
    """
    print "running switch ssm fitness simulations..."
    fitness_params_fname = input_fnames[0]
    ssm_params_fname = input_fnames[1]
    params = simulation.load_params(fitness_params_fname)
    print "FITNESS PARAMETERS: "
    print params
    raise Exception, "switch"
    model_params = simulation.load_params(ssm_params_fname)
    params.update(model_params)
    all_policies = OrderedDict()
    all_policies["Posterior predictive"] = policies.posterior_pred_policy
    all_policies["Plastic"] = policies.plastic_growth_policy
    all_policies["Random"] = policies.rand_growth_policy
    all_policies["Glucose-only"] = policies.glu_only_growth_policy
    all_policies["Posterior pred. (BH)"] = policies.bh_particle_filter_policy
    all_policies["Random (BH)"] = policies.bh_rand_growth_policy
    # fixed parameters for all simulations
    p_init_output = params["p_init_output"]
    time_obj = time_unit.Time(params["t_start"],
                              params["t_end"],
                              step_size=params["step_size"])
    # include model parameters in into simulation parameters set
    # so that the policies that need to run the model (like the
    # posterior predictive policy) can access it
    params.update(sim_params)
    # include list of policies we ran
    params["policies"] = all_policies.keys()
    # setting of probabilities for data-generating SSM model
    p_switch_to_switch = params["p_switch_to_switch"]
    p_noswitch_to_switch = params["p_noswitch_to_switch"]
    nutr_labels = params["nutr_labels"]
    nutr_growth_rates = params["nutr_growth_rates"]
    out_trans_mat1 = params["out_trans_mat1"]
    out_trans_mat2 = params["out_trans_mat2"]
    def nutrient_simulator(time_obj):
        return nutrient_simulators.ssm_nutrient_simulator(time_obj,
                                                          out_trans_mat1=out_trans_mat1,
                                                          out_trans_mat2=out_trans_mat2,
                                                          p_switch_to_switch=p_switch_to_switch,
                                                          p_noswitch_to_switch=p_noswitch_to_switch,
                                                          p_init_output=p_init_output)

    # simulate mixed sugar environment
    env_obj = \
      env.MixedDiscEnvironment(nutr_labels,
                               nutrient_simulator,
                               nutr_growth_rates=nutr_growth_rates)
    fitness_obj = fitness.FitnessSim(all_policies, env_obj, params)
    #sim_results = {}
    sim_results = fitness_obj.simulate(time_obj)
    final_results = {"sim_params": sim_params,
                     "sim_results": sim_results}
    utils.save_as_pickle(output_fname, sim_results,
                         extra={"params": params})
    
@follows(run_switch_ssm_fitness_simulations)    
@merge(run_switch_ssm_fitness_simulations,
       os.path.join(paths.SUDDEN_DATA_DIR,
                    "merged_switch_ssm_fitness_sims.data"))
def merge_switch_ssm_fitness_sims(input_fnames, output_fname):
    """
    Combine all the switch SSM fitness simulations into a single pickle file.
    """
    ### combine all the simulations into one pickle file.
    all_sim_data = OrderedDict()
    for fname in input_fnames:
        sim_name = os.path.basename(fname).split(".data")[0]
        curr_sim_data = utils.load_pickle(fname)
        all_sim_data[sim_name] = curr_sim_data
    extra = {}
    utils.save_as_pickle(output_fname, all_sim_data, extra)

@files(merge_switch_ssm_fitness_sims,
       os.path.join(PLOTS_DIR, "switch_ssm_fitness_sims.pdf"),
       "ssm_fitness_sim_plot")
def plot_switch_ssm_fitness_simulations(input_fname, plot_fname, label):
    """
    Plot switch SSM fitness simulations.
    """
    print "plotting fitness simulation for switch ssm"
    sim_info = simulation.load_data(input_fname)
    results = sim_info["data"]
    # make the plot here for three of the simulations
    sims_to_plot = ["switch_ssm_fitness_sim1",
                    "switch_ssm_fitness_sim2",
                    "switch_ssm_fitness_sim3",
                    "switch_ssm_fitness_sim4"]
    num_plots = len(sims_to_plot)
    fig = plt.figure(figsize=(10, 6))
    sns.set_style("ticks")
    ystep = 10
    yticks = np.arange(10, 40 + ystep, ystep)
    print yticks, "yticks"
    for n, sim_to_plot in enumerate(sims_to_plot):
        if sim_to_plot not in results:
            raise Exception, "No sim %s" %(sim_to_plot)
        params = results[sim_to_plot]["params"]
        params["policy_colors"] = \
          {"Random": sns.color_palette("Set1")[1],
           "Plastic": sns.color_palette("Set1")[2],
           "Posterior predictive": sns.color_palette("Set1")[0],
           "Glucose-only": sns.color_palette("Set1")[3]}
#           "Posterior pred. (BH)": "red",
#           "Random (BH)": "g"}
        plt.subplot(2, int(round(num_plots / 2.)), n + 1)
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
        model_switch_ssm.plot_fitness_sim_results(sim_results, params,
                                                  title=title,
                                                  yticks=yticks)
    fig.set_tight_layout(True)
    plt.savefig(plot_fname)


# multinutrient sudden environments
@follows(plot_sudden_markov_variable_decision_mat)
@files(None,
       os.path.join(PLOTS_DIR, "sudden_markov_multinutr_environments.pdf"),
       "fixed_decision_mat")
def plot_sudden_markov_multinutr_environments(input_fname, plot_fname, label):
    """
    Plot sudden multinutrient environments.
    """
    print "plotting: %s" %(os.path.basename(plot_fname))
    # set seed
    np.random.seed(2)
    plt.figure(figsize=(4.5, 4.05))
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
    # initial state probabilities: start with Glu
    init_probs = np.array([1., 0., 0.])
    time_obj = time_unit.Time(0, 100, step_size=1)
    num_points = len(time_obj.t)
    #plt.axvspan(a, b, color='y', alpha=0.5, lw=0)
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
    gs.update(hspace=0.6)
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
                                 annot_kws={"fontsize": 8})
        heatmap_ax.set_aspect("equal")
        ax1.tick_params(axis="both", which="major", pad=0.01, length=2.5,
                        labelsize=8)
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
            plt.xlabel("Time step", fontsize=8)
    plt.savefig(plot_fname)

def get_sudden_markov_fitness_params():
#    gluc_growth_rates = [0.3]
#    galac_growth_rates = [0.04, 0.15, 0.29]
    gluc_growth_rates = [0.7]
    galac_growth_rates = [0.175, 0.35, 0.68]
    gluc_to_gluc_probs = [0.1, 0.3, 0.5, 0.8]
    galac_to_gluc_probs = [0.1, 0.3, 0.5, 0.8]
    sim_cond = 1
    basename = "sudden_markov"
    for gluc_rate in gluc_growth_rates:
        for galac_rate in galac_growth_rates:
            for gluc_prob in gluc_to_gluc_probs:
                for galac_prob in galac_to_gluc_probs:
                    output_fname = os.path.join(paths.SUDDEN_DATA_DIR,
                                                "%s_sim%d.fitness.data" %(basename, sim_cond))
                    # simulate all parameter values
                    params = simulation.load_params(MARKOV_PARAM_FNAME)
                    params["gluc_growth_rate"] = gluc_rate
                    params["galac_growth_rate"] = galac_rate
                    params["true_gluc_to_gluc"] = gluc_prob
                    params["true_galac_to_gluc"] = galac_prob
                    yield (MARKOV_PARAM_FNAME, output_fname, params, sim_cond)
                    sim_cond += 1

@files(get_sudden_markov_fitness_params)
def simulate_sudden_markov_fitness(param_fname, output_fname, params, sim_cond):
    """
    Simulate data.
    """
    print "simulating condition %d from: %s" %(sim_cond,
                                               os.path.basename(param_fname))
    sim_results = model_sudden_markov.fitness_simulations(params)
    all_results = {"sim_results": sim_results,
                   "params": params,
                   "sim_cond": sim_cond}
    with open(output_fname, "w") as outfile:
        pickle.dump(all_results, outfile)

@merge(simulate_sudden_markov_fitness,
       os.path.join(paths.SUDDEN_DATA_DIR, "sudden_markov_fitness.all_sims.data"))
def merge_sudden_markov_fitness_sims(input_fnames, output_fname):
    """
    Merge all fitness simulation data into one pickle file.
    """
    print "merging fitness simulation files"
    all_sims = OrderedDict()
    for input_fname in input_fnames:
        data = simulation.load_data(input_fname)
        sim_cond = data["sim_cond"]
        all_sims["sim%d" %(sim_cond)] = data
    with open(output_fname, "w") as outfile:
        pickle.dump(all_sims, outfile)

@transform(simulate_sudden_markov_fitness,
           suffix(".fitness.data"),
           "_fitness.pdf",
           output_dir=PLOTS_DIR)
def plot_sudden_markov_popsize(data_fname, plot_fname):
    """
    Plot sudden Markov model population size for different growth
    policies.
    """
    print "plotting: %s" %(os.path.basename(plot_fname))
    print "  - data: %s" %(data_fname)
    sim_data = simulation.load_data(data_fname)
    sns.set_style("ticks")
    df = sim_data["sim_results"]
    params = sim_data["params"]
    plt.figure(figsize=(6, 4))
    model_sudden_markov.plot_popsize_by_policies(df, params)
    plt.savefig(plot_fname)

    
@files(BH_SIM_PARAMS_FNAMES["2_nutr"],
       os.path.join(paths.SUDDEN_DATA_DIR, "bet_hedge_sims_2_nutr.data"),
       "bet_hedge_sims_2_nutr")
@follows(plot_sudden_markov_popsize)
def run_bet_hedge_sims_2_nutr(param_fname, output_fname, label):
    """
    Running bet hedging simulations for two nutrients.
    """
    print "running bet hedge simulations (2 nutrients)..."
    # simulate two nutrient model
    params = simulation.load_params(param_fname)
    print "params: ", params
    data = {}
    extra = {"params": params}
    # run bet hedging simulation
    ###
    ### need skeleton code here
    ###
    # save results as a pickle file
    utils.save_as_pickle(output_fname, data, extra)
