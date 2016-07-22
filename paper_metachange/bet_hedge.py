##
## Bet-hedging simulations
##
import os
import sys
import time
import copy

import numpy as np

import env
import growth
import nutrients
import policies
import time_unit

from collections import OrderedDict, defaultdict

# class BetHedgeSim:
#     """
#     Bet hedge simulation.
#     """
#     def __init__(self, env_obj, policy_obj=None):
#         self.env_obj = env_obj
#         self.policy_obj = policy_obj

#     def simulate(self, init_pop_sizes, time_obj, pop_growth=False):
#         """
#         Simulate growth with bet-hedging.

#         Args:
#         -----
#         - init_pop_sizes: dictionary of initial population sizes (OrderedDict)
#         mapping nutrient (integer) to population size (integer).
#         - time_obj: time unit object.

#         Kwargs:
#         -------
#         - pop_growth: if True, use a simple population growth model, without
#         simulating individual stochastic cell growth trajectories.
#         """
#         if pop_growth:
#             results = \
#               simulate_bet_hedge_pop_growth(init_pop_sizes, time_obj,
#                                             env_obj=self.env_obj,
#                                             policy_obj=self.policy_obj)
#         else:
#             results = \
#               simulate_individual_growth(init_pop_sizes, time_obj,
#                                          env_obj=self.env_obj,
#                                          policy_obj=self.policy_obj)
#         return results


##
## old bet-hedging individual growth stochastic simulation
##
# def simulate_individual_growth(init_pop_sizes, time_obj,
#                                env_obj=None,
#                                policy_obj=None):
#     """
#     Simulate stochastic cell growth for individual cells in a population,
#     given an environment and a policy.

#     Return a list with a dictionary for each time bin, which says
#     how many cells belong to each state in that time bin.
    
#     Args:
#     -----
#     - init_pop_size: initial population size
#     - time_obj: time interval over which to simulate
#     """
#     # list of population sizes, indexed by environment
#     # state, across time
#     popsizes = []
#     times = []
#     num_timebins = len(time_obj.t)
#     prev_popsizes = copy.deepcopy(init_pop_sizes)
#     #curr_t = defaultdict(time_obj.t_start)
#     # use global timer
#     curr_t = time_obj.t_start
#     ###
#     ### TODO: fix code below so that 'growth rate' is
#     ### used appropriately only for the population type
#     ### that matches the nutrient actually used in the environment
#     ###
#     for time_ind, time_bin in enumerate(time_obj.t):
#         # for each time bin, we get the nutrient state
#         # and corresponding growth rate using the policy
#         result = policy_obj.run(time_obj, env_obj)
#         growth_rate = result["growth_rates"][time_ind]
#         next_nutr_state = int(result["nutrient_states"][time_ind])
#         # dictionary representing current population size
#         curr_popsizes = copy.deepcopy(prev_popsizes)
#         # all of the nutrient states currently existing
#         # in the population
#         all_nutr_states = [k for k in curr_popsizes \
#                            if curr_popsizes[k] > 0]
#         num_nutr_states = len(all_nutr_states)
#         while curr_t < time_bin:
#             ## simulate growth here separately for population
#             ## (e.g. for glucose population, galactose population)
#             for curr_nutr in all_nutr_states:
#                 # stochastically simulate cell division
#                 if (growth_rate > 0) and (curr_popsizes[curr_nutr] > 0):
#                     new_rate = 1/float(growth_rate * curr_popsizes[curr_nutr])
#                     div_time = np.random.exponential(scale=new_rate)
#                     # advance time
#                     curr_t += div_time
#                     # increase population size and choose the next nutrient state
#                     curr_popsizes[next_nutr_state] += 1
#                 else:
#                     # if the growth rate is zero for this time bin,
#                     # simply advance to next time bin
#                     curr_t = time_bin
#         # accumulate the current population size
#         # for the time bin
#         times.append(time_bin)
#         popsizes.append(curr_popsizes)
#         # set previous population size to current population
#         # size for next time point
#         prev_popsizes.update(curr_popsizes)
#     # check that we have population sizes for each time point
#     assert (len(popsizes) == len(time_obj.t)), "Population sizes " \
#            "do not match time interval."
#     return popsizes


def run_bet_hedge_sim():
    # simulate environment
    # use posterior bet hedging
    # compare to Jablonka's strategy
    nutr_labels = ["glucose", "galactose"]
    env_obj = \
      env.MixedDiscEnvironment(nutr_labels, nutrients.discrete_markov_glucose,
                               nutr_growth_rates=[0.3, 0.3/4],
                               mismatch_growth_rate=0.0)
#    step_size = 0.5
#    time_obj = time_unit.Time(0, 10, step_size)
    time_obj = time_unit.Time(0, 10, 1)
    env_obj.simulate(time_obj)
#    policy_obj = \
#      growth.GrowthPolicy(policy_func=policies.ind_random_growth_policy)
    policy_obj = \
      growth.GrowthPolicy(policy_func=policies.ind_glu_only_growth_policy)
    bh = BetHedgeSim(env_obj, policy_obj=policy_obj)
    init_pop_sizes = OrderedDict()
    init_pop_sizes[0] = 1
    init_pop_sizes[1] = 0
    bh.simulate(init_pop_sizes, time_obj)

def main():
    pass
    #run_bet_hedge_sim()

if __name__ == "__main__":
    main()

    
