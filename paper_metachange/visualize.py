##
## Visualization code for environments
## 
import os
import sys
import time

import numpy as np

import matplotlib.pylab as plt
import seaborn as sns


def plot_env(env, growth_data=None, log_pop_size=True):
    """
    Plot environment and the population growth (if given).
    """
    if growth_data is None:
        df = env.as_df(melted=True)
        g = sns.factorplot(x="index", y="value",
                           hue="nutrient", data=df)
        return g
    # get un-melted dataframe
    df = env.as_df(melted=False)
    df["t"] = growth_data["t"].t
    plt.subplot(2, 1, 1)
    df["pop_size"] = np.exp(growth_data["log_pop_size"])
    for nutr in env.nutrs:
        plt.plot(df["t"], df[nutr], label=nutr)
    plt.legend()
    plt.xlabel("Time")
    plt.subplot(2, 1, 2)
    plt.xlabel("Time")
    if log_pop_size:
        plt.plot(df["t"], np.log2(df["pop_size"]))
        plt.ylabel("Pop. size (log$_\mathrm{2}$)")
    else:
        plt.plot(df["t"], df["pop_size"])
        plt.ylabel("Pop. size")
    
    

