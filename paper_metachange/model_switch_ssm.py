##
## Utilities related to analysis that uses switching SSMs
##
import os
import sys
import time

from collections import OrderedDict

import numpy as np
import pandas

import matplotlib.pylab as plt
import seaborn as sns

import particlefever
import particlefever.particle_filter as particle_filter
import particlefever.switch_ssm as switch_ssm

import growth
import policies
import time_unit
import prob_utils
import fitness
import env


def plot_fitness_sim_results(df, params,
                             ymin=None,
                             ymax=None,
                             yticks=None,
                             title=None,
                             x_step=10):
    sns.set_style("ticks")
   # plot population size
    popsizes = fitness.str_popsizes_to_array(df["log_pop_sizes"])
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
                        legend=False)
        for policy_num, policy in enumerate(policy_colors):
            error_df = summary_df[summary_df["policy"] == policy]
            c = policy_colors[policy]
            assert (len(error_df["t"]) == len(time_obj.t) == \
                    len(error_df["log2_pop_size"]["mean"]))
        plt.xlabel("Time step", fontsize=10)
        plt.ylabel("Pop. size ($\log_{2}$)", fontsize=10)
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
        plt.xticks(range(int(time_obj.t.min()), int(time_obj.t.max()) + x_step,
                         x_step),
                   fontsize=8)
        if yticks is not None:
            plt.yticks(yticks, fontsize=8)
            plt.ylim(yticks[0], yticks[-1])
        sns.despine(trim=True, offset=2*time_obj.step_size)
        plt.tick_params(axis='both', which='major', labelsize=8,
                        pad=2)
    # make plot
    plot_pop_size_across_time(params)
    # plot population size
    # popsize_df = {}
    # popsize_df["policy"] = final_df["policy"]
    # popsize_df["log2_final_popsize"] = final_df["log2_pop_size"]["mean"]
    # popsize_df["log2_final_popsize_std"] = final_df["log2_pop_size"]["std"]
    # popsize_df = pandas.DataFrame(popsize_df)
    # x_axis = range(len(popsize_df["policy"]))
    # c = [policy_colors[p] for p in popsize_df["policy"]]
    # # get dataframe with final data points
    # df_to_plot = df[df["t"] == time_obj.t[-1]]


def main():
    pass

if __name__ == "__main__":
    main()
