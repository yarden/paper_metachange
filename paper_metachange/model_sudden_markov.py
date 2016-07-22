##
## Model of sudden nutrient switches in a Markov chain
##
import os
import sys
import time

import pandas
import numpy as np
import scipy
import scipy.stats

import matplotlib
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
sns.set_style("white")

from collections import OrderedDict

import plot_utils
import prob_utils
import paths
import growth
import fitness
import time_unit
import env
import nutrients
import policies
import simulation as sim

def simulate_nutrient_trans(num_points,
                            p_gluc_to_gluc,
                            p_galac_to_gluc,
                            p_carbon_state=0.5,
                            init_state=None,
                            num_unobserved=0):
    """
    Simulate nutrient tranisitons, assuming
    gluc is True, galac is False.
    """
    if init_state is None:
        init_state = (np.random.rand() <= p_carbon_state)
    data = [init_state]
    if num_points == 1:
        return data
    points = range(num_points) 
    for n in points[1:]:
        if data[n-1] == True:
            # probability of transitioning from gluc to gluc
            next_state = (np.random.rand() <= p_gluc_to_gluc)
        else:
            next_state = (np.random.rand() <= p_galac_to_gluc)
        data.append(next_state)
    # if asked, add unobserved data
    for n in range(num_unobserved):
        data.append(np.nan)
    return data

def get_max_prob_state(prev_value, p_gluc_to_gluc, p_galac_to_gluc,
                       gluc_val=True,
                       galac_val=False):
    """
    Select maximum probability state.
    """
    flip = np.random.rand()
    state = None
    if prev_value == gluc_val:
        # previous value was glucose
        state = (flip <= p_gluc_to_gluc)
    elif prev_value == galac_val:
        # previous value was galactose
        state = (flip <= p_galac_to_gluc)
    else:
        raise Exception, "Unhandled value: %s" %(str(prev_value))
    return state
    
def predict_growth_policy(time_obj, env, params):
    """
    maximum probability growth policy.

    if glu->glu > 0.5, then always glu
    """
    return adaptive_growth_policy(time_obj, env, params, "predict")

def smooth_growth_policy(time_obj, env, params):
    """
    maximum probability growth policy.

    if glu->glu > 0.5, then always glu
    """
    return adaptive_growth_policy(time_obj, env, params, "smooth")


def fitness_simulations(params):
    """
    Fitness simulations.
    """
    true_gluc_to_gluc = params["true_gluc_to_gluc"]
    true_galac_to_gluc = params["true_galac_to_gluc"]
    init_pop_size = params["init_pop_size"]
    num_sim_iters = params["num_sim_iters"]
    step_size = params["step_size"]
    t_start = params["time_start"]
    t_end = params["time_end"]
    mismatch_growth_rate = params["mismatch_growth_rate"]
    nutr_growth_rates = [params["gluc_growth_rate"],
                         params["galac_growth_rate"]]
    time_obj = time_unit.Time(t_start, t_end, step_size=step_size)
    num_points = len(time_obj.t)
    data = \
      simulate_nutrient_trans(num_points,
                              true_gluc_to_gluc,
                              true_galac_to_gluc,
                              init_state=True)
    # environment simulator
    def gluc_galac_simulator(time_obj):
        return nutrients.discrete_markov_glucose_galactose(time_obj,
                                                           true_gluc_to_gluc,
                                                           true_galac_to_gluc)
    print "running %d simulation iterations" %(num_sim_iters)
    sim_results = []
    assert (len(time_obj.t) == num_points), "Unequal number of time slices " \
           "and data points."
    all_policies = OrderedDict()
    all_policies["Glucose-only"] = policies.glu_only_growth_policy
    all_policies["Adaptive"] = policies.true_growth_policy
    all_policies["Random"] = policies.rand_growth_policy
    nutr_labels = ["glucose", "galactose"]
    for sim_num in range(1, num_sim_iters + 1):
        # simulate mixed sugar environment
        env_obj = \
          env.MixedDiscEnvironment(nutr_labels, gluc_galac_simulator,
                                   nutr_growth_rates=nutr_growth_rates,
                                   mismatch_growth_rate=mismatch_growth_rate)
        env_obj.simulate(time_obj)
        # get growth rates/behavior for each policy
        for curr_policy in all_policies:
            df = {"true_gluc_to_gluc": params["true_gluc_to_gluc"],
                  "true_galac_to_gluc": params["true_galac_to_gluc"],
                  "sim_num": sim_num,
                  "t": time_obj.t}
            ## TODO: refactor this into 'run_policy_on_env' function
            policy_f = all_policies[curr_policy]
            policy_obj = policies.GrowthPolicy(policy_func=policy_f,
                                               policy_params=params)
            growth_obj = growth.Growth(init_pop_size,
                                       env_obj=env_obj,
                                       policy_obj=policy_obj)
            policy_results = growth_obj.simulate(time_obj)
            policy_df = pandas.DataFrame(df)
            policy_df["policy"] = curr_policy
            policy_df["growth_rates"] = policy_results["growth_rates"]
            # convert population sizes to single (total) population size
            popsizes = policy_results["log_pop_sizes"]
            policy_df["log2_pop_size"] = np.log2(np.exp(popsizes).sum(axis=1))
            sim_results.append(policy_df)
    sim_results = pandas.concat(sim_results)
    return sim_results

def plot_fitness_simulations(df, params):
    plt.figure()
    plot_popsize_by_policies(df, params)
    plt.show()

def plot_popsize_by_policies(df, params, ymin=None, ymax=None):
    sns.set_style("ticks")
    # plot population size
    grouped = df.groupby(["t", "policy"], as_index=False)
    summary_df = grouped.agg({"log2_pop_size": [np.mean, np.std],
                              "growth_rates": [np.mean, np.std]})
    rand_summary = summary_df[summary_df["policy"] == "Random"]
    policy_colors = {"Glucose-only": sns.color_palette("Set1")[2],
                     "Random": sns.color_palette("Set1")[1],
                     "Adaptive": sns.color_palette("Set1")[0]}
    time_obj = time_unit.Time(params["time_start"],
                              params["time_end"],
                              step_size=params["step_size"])
    step_size = params["step_size"]
    ## plot population size across time
    def plot_pop_size_across_time(params,
                                  ymin=ymin,
                                  ymax=ymax):
        offset = step_size / 250.
        x_offsets = [0, offset, 2*offset]
        num_xticks = 11
        ax = sns.tsplot(time="t", value="log2_pop_size", unit="sim_num",
                        condition="policy", color=policy_colors,
                        err_style=None,
                        data=df)
        for policy_num, policy in enumerate(policy_colors):
            error_df = summary_df[summary_df["policy"] == policy]
            c = policy_colors[policy]
            assert (len(error_df["t"]) == len(time_obj.t) == \
                    len(error_df["log2_pop_size"]["mean"]))
            plt.errorbar(error_df["t"] + x_offsets[policy_num],
                         error_df["log2_pop_size"]["mean"],
                         yerr=error_df["log2_pop_size"]["std"].values,
                         color=c,
                         linestyle="none",
                         marker="o",
                         capsize=2,
                         capthick=1,
                         clip_on=False)
        plt.xlabel(r"$t$")
        plt.ylabel("Pop. size ($\log_{2}$)")
        #
        #\theta_{\mathsf{Gal}\rightarrow\mathsf{Glu}}
        print "PARAMS: ", params
        plt.title(r"$P_{0} = %d$, $\theta_{\mathsf{Glu}\rightarrow \mathsf{Glu}} = %.2f$, " \
                  r"$\theta_{\mathsf{Gal}\rightarrow \mathsf{Glu}} = %.2f$, " \
                  r"$\mu_{\mathsf{Glu}} = %.2f, \mu_{\mathsf{Gal}} = %.2f$, " \
                  r"$\mu_{\mathsf{Mis}} = %.2f$, lag = %d, " \
                  r"%d iters" %(sum(params["init_pop_size"]),
                                params["true_gluc_to_gluc"],
                                params["true_galac_to_gluc"],
                                params["gluc_growth_rate"],
                                params["galac_growth_rate"],
                                params["mismatch_growth_rate"],
                                params["decision_lag_time"],
                                params["num_sim_iters"]),
                 fontsize=8)
        c = 0.5
        plt.xlim([min(df["t"]) - c, max(df["t"]) + c])
        if ymin is None:
            ymin = int(np.log2(sum(params["init_pop_size"])))
        if ymax is None:
            ymax = int(error_df["log2_pop_size"]["mean"].max() + \
                       error_df["log2_pop_size"]["std"].max()) + 1
        plt.ylim([ymin, ymax])
        plt.xlim([time_obj.t.min(),
                  time_obj.t.max()])
#        plt.yticks(range(ymin, ymax + 1))
        plt.xticks(np.linspace(time_obj.t.min(),
                               time_obj.t.max(),
                               num_xticks))
        sns.despine(trim=True, offset=time_obj.step_size*2)
    # make plot
    plot_pop_size_across_time(params)
    plt.tight_layout()


def expected_reward(policy, params):
    """
    Implement expected rewards for different strategies
    in Markov chain. Assumes the growth rate for nutrient state
    mismatched to environment is zero.
    """
    p_gluc_to_gluc = params["p_gluc_to_gluc"]
    p_galac_to_gluc = params["p_galac_to_gluc"]
    gluc_growth_rate = params["gluc_growth_rate"]
    galac_growth_rate = params["galac_growth_rate"]
    reward = None
    def all_gluc_reward():
        return (gluc_growth_rate * p_gluc_to_gluc) + \
               (gluc_growth_rate * p_galac_to_gluc)
    def all_galac_reward():
        return (galac_growth_rate * (1 - p_gluc_to_gluc)) + \
               (galac_growth_rate * (1 - p_galac_to_gluc))
    if policy == "all_gluc":
        reward = all_gluc_reward()
    elif policy == "all_galac":
        reward = all_galac_reward()
    elif policy == "adaptive":
        # adaptive policy uses the probabilities of
        # glucose and galactose to adapt
        if (p_gluc_to_gluc >= 0.5) and (p_galac_to_gluc >= 0.5):
            # same as all_glucose reward
            reward = all_gluc_reward()
        elif (p_gluc_to_gluc >= 0.5) and (p_galac_to_gluc < 0.5):
            reward = (gluc_growth_rate * p_gluc_to_gluc) + \
                     (galac_growth_rate * (1 - p_galac_to_gluc))
        elif (p_gluc_to_gluc < 0.5) and (p_galac_to_gluc >= 0.5):
            reward = (galac_growth_rate * (1 - p_gluc_to_gluc)) + \
                     (gluc_growth_rate * p_galac_to_gluc)
        elif (p_gluc_to_gluc < 0.5) and (p_galac_to_gluc < 0.5):
            reward = (galac_growth_rate * (1 - p_gluc_to_gluc)) + \
                     (galac_growth_rate * (1 - p_galac_to_gluc))
        else:
            raise Exception, "Unhandled reward case."
    else:
        raise Exception, "Unknown policy."
    return reward
        

def decision_ratio(policy1, policy2, params,
                   policies=["all_gluc",
                             "all_galac",
                             "adaptive"]):
    """
    Decision ratio in Markov chain.
    """
    if policy1 not in policies:
        raise Exception, "Unknown policy %s" %(policy1)
    if policy2 not in policies:
        raise Exception, "Unknown policy %s" %(policy2)
    policy1_reward = expected_reward(policy1, params)
    policy2_reward = expected_reward(policy2, params)
    if policy2_reward == 0:
        raise Exception, "Zero reward policy."
    ratio = policy1_reward / float(policy2_reward)
    return ratio


def test_decision_analysis():
    params = {"gluc_growth_rate": 0.3,
              "galac_growth_rate": 0.3/7.}
    # both probabilities > 0.5
    probs = np.linspace(0.5, 1, 11)
    policy1 = "adaptive"
    policy2 = "all_gluc"
    for p1 in probs:
        for p2 in probs:
            params["p_gluc_to_gluc"] = p1
            params["p_galac_to_gluc"] = p2
            ratio = decision_ratio(policy1, policy2, params)
            assert (ratio == 1), "Error"

def get_decision_ratio_mat(probs, params,
                           discretize=True):
    mat = []
    for p_gluc_to_gluc in probs:
        params["p_gluc_to_gluc"] = p_gluc_to_gluc
        ratios = []
        for p_galac_to_gluc in probs:
            params["p_galac_to_gluc"] = p_galac_to_gluc
            curr_ratio = decision_ratio("adaptive", "all_gluc", params)
            ratios.append(curr_ratio)
        mat.append(ratios)
    mat = np.array(mat)
    if discretize:
        # values around 1 are set to 0 (equal)
        mat[mat == 1] = 0
        # values greater than 1 get set to 1 (adaptive policy wins)
        mat[mat > 1] = 1
        # values less than 1 get set to -1 (all gluc wins)
        mat[np.where(np.logical_and(mat > 0, mat < 1))] = -1
    return mat


def plot_decision_mat(params,
                      palette="seismic",
                      discretize=False,
                      show_x_label=False,
                      show_y_label=False,
                      show_colorbar_label=False):
    from matplotlib.colors import LogNorm, BoundaryNorm, SymLogNorm
    probs = np.linspace(0.001, 1, 101)
    import matplotlib.colors as mcolors
    if discretize:
        cmap, norm = \
          mcolors.from_levels_and_colors([-1, 0, 1, 5],
          sns.color_palette(palette)[0:3][::-1])

    else:
        cmap = plt.cm.get_cmap(palette)# + "_r")
    galac_rate = params["galac_growth_rate"]
    mat = get_decision_ratio_mat(probs, params, discretize=discretize)
    sns.set_style("white")
    # Full quantitative view
    log2_mat = np.log2(mat)
    log_min_val = -3
    log_max_val = 3
    # clip values
    log2_mat = np.clip(log2_mat, a_min=log_min_val, a_max=log_max_val)
    # plot heatmap
    heatmap = plt.pcolormesh(log2_mat, cmap=cmap, linewidth=0,
                             vmin=log_min_val, vmax=log_max_val)
    heatmap_ax = heatmap.axes
    heatmap_ax.set_aspect("equal")
    tick_space = 10
    xticks = np.arange(len(probs)) + 0.5
    yticks = xticks
    xlabels = [""] * len(xticks)
    ylabels = [""] * len(yticks)
    ##
    ## TODO: change this to have fewer x/y ticks in heatmaps
    ##
    for n in range(0, len(xticks), tick_space):
        xlabels[n] = "%.2f" %(probs[n])
        ylabels[n] = "%.2f" %(probs[n])
    plt.xticks(xticks, rotation="vertical")
    plt.yticks(yticks)
    plt.gca().set_xticklabels(xlabels, fontsize=7)
    plt.gca().set_yticklabels(ylabels, fontsize=7)
    plt.xlim(xticks[0] - 0.5, xticks[-1] + 0.5)
    plt.ylim(yticks[0] - 0.5, yticks[-1] + 0.5)
    heatmap_ax.tick_params(axis='both', which='major', pad=4)
    if show_x_label:
        plt.xlabel(r"$\theta_{\mathsf{Gal}\rightarrow\mathsf{Glu}}$")
    if show_y_label:
        plt.ylabel(r"$\theta_{\mathsf{Glu}\rightarrow\mathsf{Glu}}$")
    plt.title(r"$\mu_\mathsf{Glu} = %.2f, \mu_\mathsf{Gal} = %.2f$" \
              %(params["gluc_growth_rate"],
                params["galac_growth_rate"]),
              fontsize=8)
    # plt.title(r"$\mu_\mathsf{Glu} = %.2f, \mu_\mathsf{Gal} = %.2f, " \
    #            "\mu_\mathsf{Mis} = %.2f$" \
    #           %(params["gluc_growth_rate"],
    #             params["galac_growth_rate"],
    #             params["mismatch_growth_rate"]),
    #           fontsize=8)
    # plot colorbar
    sns.set_style("ticks")
    divider = make_axes_locatable(heatmap_ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if discretize:
        p = plt.colorbar()
        p.set_ticks([-1, 0, 1])
        #p.set_ticklabels(["Glucose only", "Equal", "Adaptive"])
    else:
        # set ticks for non-discretized colorbar
        tick_space = 1
        log_ticks = np.arange(log_min_val, log_max_val + tick_space, tick_space)
        p = plt.colorbar(ticks=log_ticks, cax=cax)
    if discretize:
        p.set_label("Optimal policy")
    else:
        if show_colorbar_label:
           p.set_label("Posterior pred. / Glu-only\n growth rate ($\\log_2$)",
                       fontsize=8)
    p.ax.tick_params(labelsize=7)
    ### TODO: figure out how to correctly set tick padding here
    p.ax.tick_params(axis="both", which="both", pad=-5, length=3)
    cax.tick_params(axis="both", which="both", pad=-5, length=3)
#        p.set_label(r"Adaptive / glucose-only fitness ($\log_2$)",
#                    fontsize=10)
      
        
    sns.set_style("white")

def plot_decision_analysis(params):
    """
    Plot decision analysis.
    """
    # calculate decision matrix
    # reverse entries so that y-axis is fixed
    ## plot for fixed rates
    plot_fname = os.path.join(paths.PLOTS_DIR, "decision_fixed_rates.pdf")
    # currently, we do not handle cases where \mu_mis isn't 0 (i.e. when
    # growth rate when cell state is mismatched to environment)
    # as sanity check, look that we're not passed this value in the parameters
    if not np.allclose(params["mismatch_growth_rate"], 0):
        print "WARNING: passed mismatch_growth_rate != 0. Decision analysis " \
              "matrix does not handle this."
    fig = plt.figure(figsize=(5, 4))
    plot_decision_mat(params)
    plt.tight_layout()
    plt.savefig(plot_fname)
