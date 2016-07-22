##
## Nutrient datasets
##
import os
import sys
import time

import pandas

import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import scipy
import scipy.stats

import monod
import units

##
## Data from Boer et. al. (Bostein, Rabinowitz)
## paper
##
boer_data = \
  pandas.DataFrame({"avg_dil_rate_per_hr":
                   [0.058, 0.108, 0.161, 0.238, 0.311],
                   "sd_dil_rate_per_hr":
                   [0.001, 0.011, 0.001, 0.003, 0.004],
                   "avg_extracellular_gluc_mM":
                   [0.104, 0.118, 0.149, 0.252, 1.133],
                   "sd_extracellular_gluc_mM":
                   [0.022, 0.015, 0.001, 0.016, 0.020]})

# Fit Monod equation to Boer et. al. data
S_num_bins = 50
S = \
  np.linspace(0, 2*boer_data["avg_extracellular_gluc_mM"].max(),
              S_num_bins)
params = monod.fit_monod(boer_data["avg_extracellular_gluc_mM"],
                         boer_data["avg_dil_rate_per_hr"])
monod_vals = monod.monod_eq(S, params["mu_max"], params["K_s"])

def get_growth_rate_from_gluc(gluc_mM):
    """
    Return growth rate (dilution rate) given
    glucose level (in mM) based on Boer et. al. (2010)
    paper.
    """
    return monod_eq(gluc_mM, params["mu_max"], params["K_s"])

def plot_nutrient_data():
    sns.set_style("white")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(boer_data["avg_extracellular_gluc_mM"],
                boer_data["avg_dil_rate_per_hr"],
                yerr=boer_data["sd_dil_rate_per_hr"],
                fmt="o")
    ax.plot(S, monod_vals, "r-")
    plt.xlabel("Avg. extracellular glucose (mM)")
    plt.ylabel("Avg. dilution rate per hr")
    plt.title(r"Fit from Boer et. al. (2010), " \
              "$\mu_{\mathregular{max}} = %.2f, " \
              "K_{s} = %.2f$" %(params["mu_max"],
                                params["K_s"]))
    plt.show()

if __name__ == "__main__":
    plot_nutrient_data()
