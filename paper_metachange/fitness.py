##
## Fitness simulator.
##
## Takes the growth results of different policies.
##
import os
import sys
import time

import pandas
import numpy as np

import env
import growth
import policies

def popsizes_as_str(popsizes, delim=","):
    """
    Convert population sizes to strings.
    
    population sizes are an array of K x N elements
    where N is number of nutrients and K is an arbitrary
    length of the simulation time.
    """
    str_popsizes = []
    for curr_sizes in popsizes:
        curr_str = ["%.1f" %(s) for s in curr_sizes]
        str_popsizes.append(delim.join(curr_str))
    return str_popsizes

def str_popsizes_to_array(str_popsizes, delim=","):
    """
    Convert list of comma-separated population sizes
    (e.g. ['10,1', '20,2', ...]) to an array of floats
    (e.g. [[10, 1], [20, 2], ...]).
    """
    popsizes = []
    for curr_sizes in str_popsizes:
        float_sizes = map(float, curr_sizes.split(delim))
        popsizes.append(float_sizes)
    return np.array(popsizes)

class FitnessSim:
    def __init__(self, all_policies, env_obj, params):
        """
        Args:
        -----
        - all_policies: dictionary (OrderedDict) mapping policy labels
          to their functions.
        - nutrient_simulator: function that generates nutrient simulation
        (to be passed as argument to Environment
        - params: fitness simulation parameters.
        """
        self.all_policies = all_policies
        self.env_obj = env_obj
        self.params = params

    def simulate(self, time_obj):
        """
        Fitness simulations.
        """
        init_pop_sizes = self.params["init_pop_sizes"]
        num_sim_iters = self.params["num_sim_iters"]
        num_points = len(time_obj.t)
        print "running %d simulation iterations" %(num_sim_iters)
        t1 = time.time()
        sim_results = []
        assert (len(time_obj.t) == num_points), \
          "Unequal number of time slices and data points."
        nutr_labels = ["glucose", "galactose"]
        nutr_labels = self.params["nutr_labels"]
        nutr_growth_rates = self.params["nutr_growth_rates"]
        for sim_num in range(1, num_sim_iters + 1):
            self.env_obj.simulate(time_obj)
            # get growth rates/behavior for each policy
            for curr_policy in self.all_policies:
                df = {"sim_num": sim_num,
                      "t": time_obj.t}
                policy_f = self.all_policies[curr_policy]
                t1 = time.time()
                policy_obj = policies.GrowthPolicy(policy_func=policy_f,
                                                   policy_params=self.params)
                growth_obj = growth.Growth(init_pop_sizes,
                                           env_obj=self.env_obj,
                                           policy_obj=policy_obj)
                policy_results = growth_obj.simulate(time_obj)
                policy_df = pandas.DataFrame(df)
                policy_df["policy"] = curr_policy
                policy_df["growth_rates"] = policy_results["growth_rates"]
                policy_df["pop_sizes"] = \
                  popsizes_as_str(policy_results["pop_sizes"])
                policy_df["log_pop_sizes"] = \
                  popsizes_as_str(np.log(policy_results["pop_sizes"]))
                t2 = time.time()
                sim_results.append(policy_df)
        sim_results = pandas.concat(sim_results)
        print "simulations took %.2f mins" %((t2 - t1) / 60.)
        # print "SIM RESULTS: "
        # print " -- " * 10
        # print sim_results
        return sim_results
