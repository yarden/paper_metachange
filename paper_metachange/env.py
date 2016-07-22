##
## Implementation of environments
##
##
import os
import sys
import time

import numpy as np
import pandas

import matplotlib.pylab as plt
import seaborn as sns

import visualize as viz
import growth
import nutrients
import time_unit

from collections import OrderedDict

class Environment:
    """
    Environment for growth.
    """
    def __init__(self, nutrs):
        """
        - nutrs: mapping from nutrient name to function that simulates it
          if "mixed" is given, then environment is assumed to be a mixed
          discrete environment.
        """
        self.nutrs = nutrs
        # nutrient history
        self.hist = OrderedDict()
        self.time_obj = None
        self.num_nutrs = len(self.nutrs)

    def simulate(self, time_obj):
        self.time_obj = time_obj
        num_nutrs = len(self.nutrs)
        for curr_nutr in self.nutrs:
            nutr_simulator = self.nutrs[curr_nutr]
            data = nutr_simulator(self.time_obj)
            self.hist[curr_nutr] = data

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Env(nutrs=%s, hist=%s)" %(self.nutrs, self.hist)


class MixedDiscEnvironment:
    """
    Mixed discrete environment for growth, where nutrients
    are mutually exclusive.
    """
    def __init__(self, nutr_labels, nutr_simulator,
                 nutr_growth_rates=[],
                 mismatch_growth_rate=0.0):
        # nutrient labels: list of nutrients, e.g.
        # ["glucose", "galactose"] where the 0-based
        # position of the nutrient is assumed to be its value
        self.nutr_labels = nutr_labels
        self.num_nutrs = len(self.nutr_labels)
        # mapping from nutrient label to its integer value
        # e.g. {"glucose": 0, "galactose": 1}.
        # the simulator uses the integer values
        self.nutr_label_to_val = \
          dict([(self.nutr_labels[n], n) for n in xrange(self.num_nutrs)])
        # inverse mapping: from nutrient integer value to label
        self.val_to_nutr_label = \
          dict([(self.nutr_label_to_val[k], k) \
                for k in self.nutr_label_to_val])
        # nutrient history simulator
        self.nutr_simulator = nutr_simulator
        # nutrient history
        self.hist = []
        # growth rate for each nutrient as a list, e.g.:
        #   [0.3, 0.3/7]
        # where the order of nutrients is determined by 'nutr_labels'
        self.nutr_growth_rates = nutr_growth_rates
        # growth rate when cells's nutrient state doesn't match the
        # environment's nutrient state
        self.mismatch_growth_rate = mismatch_growth_rate
        self.time_obj = None
        assert (len(self.nutr_growth_rates) == len(self.nutr_labels)), \
          "Need same number of growth rates as nutrient labels."

    def get_growth_rate(nutr_name):
        """
        return growth rate of nutrient by its string name.
        """
        if nutr_name not in self.nutr_labels:
            raise Exception, "No nutrient %s" %(nutr_name)
        nutr_ind = self.nutr_labels.index(nutr_name)
        return self.nutr_growth_rates[nutr_ind]

    def simulate(self, time_obj):
        if len(self.nutr_growth_rates) == 0:
            raise Exception, \
              "Need nutrient growth rates to simulate environment."
        self.time_obj = time_obj
        for curr_nutr in self.nutr_labels:
            data = self.nutr_simulator(self.time_obj)
            self.hist = data

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Env(nutrs=%s, hist=%s)" %(self.nutrs, self.hist)

##
## Different glucose policies
##
def glucose_growth_policy1(time_obj, env, growth_obj=None,
                           params=None):
    """
    Glucose growth policy #1:
    Adjust growth rate as a (bounded) linear function of glucose
    abundance.

    Args:
    - time_obj: time (binned) to behave in
    - env: environment object

    Kwargs:
    - growth_obj: growth object (seeing population growth)
    """
    if "glucose" not in env.hist:
        raise Exception, "Glucose policy needs glucose in environment."
    slope = 0.3
    max_rate = 0.35
    glucose_hist = np.array(env.hist["glucose"])
    growth_rates = map(lambda x: min(max_rate, x), glucose_hist * slope)
    result = {"growth_rates": growth_rates}
    return result

def glucose_growth_policy2(time_obj, env, growth_obj=None,
                           gluc_min_thresh=1):
    """
    Glucose growth policy #2:
    Adjust growth as a (bounded) linear function of glucose
    abundance, *assuming* glucose is above a threshold.
    """
    if "glucose" not in env.hist:
        raise Exception, "Glucose policy needs glucose in environment."
    slope = 0.3
    max_rate = 0.35
    glucose_hist = np.array(env.hist["glucose"])
    def gluc_to_rate(x):
        if x < gluc_min_thresh:
            return 0 
        return min(max_rate, x * slope)
    growth_rates = map(gluc_to_rate, glucose_hist)
    result = {"growth_rates": growth_rates}
    return result


##
## TODO: For now, simulate environment first, then
## simulate growth policy. Change this in future to be
## so that growth is part of environment, so that you
## can have feedback: growth affecting the environment.
##
# Plot example environment along with growth population
#env = Environment({"glucose": nutrients.noisy_cont_glucose})
#env = Environment({"glucose": nutrients.discrete_markov_glucose})
if __name__ == "__main__":
    env = Environment({"glucose": nutrients.discrete_markov_glucose})
    init_pop_size = 10
    time_interval = 100
    time_obj = time_unit.Time(0, time_interval, 0.2)
    # simulate environment
    env.simulate(time_obj)
    # make growth policy
    policy_obj = growth.GrowthPolicy(policy_func=glucose_growth_policy1)
    #policy_obj = growth.GrowthPolicy(policy_func=glucose_growth_policy2)
    # make growth environment
    growth_obj = growth.Growth(init_pop_size,
                               env=env,
                               policy=policy_obj)
    # simulate growth
    growth_data = growth_obj.simulate(time_obj)
    sns.set_style("ticks")
    g = viz.plot_env(env, growth_data=growth_data)
    plt.tight_layout()
    sns.despine()
    plt.show()

    #growth_data = growth.exp_growth(init_pop_size,
    #                                growth_rate_per_hr,
    #                                time_interval)
    #df = env.as_df()

    ## TODO: make an interface for simulating a nutrient
    ## a nutrient is a function that, when given a set of previous
    ## inputs, draws another input
