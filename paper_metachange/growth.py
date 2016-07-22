##
## Growth simulator
##
import os
import sys
import time

from collections import OrderedDict, defaultdict

import numpy as np
import scipy
import scipy.stats
import matplotlib.pylab as plt
import seaborn as sns
import pandas

import time_unit
import policies
import copy

def exp_growth(init_pop_size, growth_rate_per_hr, time_interval,
               time_num_bins=20, time_start=0):
    """
    Simulate exponential growth.

    P(t) = C*2^(rt)

    logP(t) = log(C) + rt*log(2)

    where C is initial population size, r is growth rate,
    and t is time interval. Return log population size across time.

    Args:
    - init_pop_size: initial population size
    - growth_rate_per_hr: growth rate per hour, e.g. 0.3/hr
    - time_interval: interval of time, e.g. 3 hours

    Kwargs:
    - time_num_bins: number of time bins
    - time_start: time to start at (0 by default)
    """
    time_axis = np.linspace(time_start, time_interval, time_num_bins)
    log_pop_size = \
      np.log(init_pop_size) + (growth_rate_per_hr * time_axis)*np.log(2)
    return {"t": time_axis, "log_pop_size": log_pop_size}


def simulate_growth(init_pop_size, time_obj, growth_rates):
    """
    Simulate population growth given growth rates for each
    time bin. Returns log population size (natural log).
    
    Args:
    - init_pop_size:
    - time_axis: binned time axis
    - growth_rates: growth rates for each bin (same length as time_axis)
    """
    log_pop_size = [np.log(init_pop_size)]
    t_start = time_obj.t[0]
    curr_pop_size = log_pop_size[0]
    for n in xrange(1, time_obj.num_steps):
        # growth rate for 0 through first time interval
        t_end = time_obj.t[n]
        t_duration = t_end - t_start 
        # current bin's growth rate
        curr_rate = growth_rates[n]
        # write as log
        curr_pop_size += ((t_duration * curr_rate) * np.log(2))
        log_pop_size.append(curr_pop_size)
        t_start = t_end
    return log_pop_size


def simulate_pop_growth(init_pop_sizes, time_obj,
                        env_obj=None,
                        policy_obj=None):
    """
    Simulate deterministic population growth (with optional bet-hedging
    for nutrient states), given an environment and a policy.

    The growth process is:

    1. simulate growth of initial population types (e.g. glucose, galactose),
    according to their respective growth rates, for the current time bin.
    2. at end of time bin, use policy to determine what fraction of 
    the cells should be of each nutrient state (glucose, galactose) etc.
    3. continue process for next time bin.

    Note that the lag between exposure to environment and decision time
    is encoded in the policy. The policy should impose the lag, so that
    cells don't change nutrient states too quickly. This function has
    no notion of lag; it simply takes the nutrient state returned
    by the policy.
    """
    # list of population sizes, indexed by environment
    # state, across time
    popsizes = []
    times = []
    num_timebins = len(time_obj.t)
    # previous population size takes copy of initial
    # population sizes list
    prev_popsizes = list(init_pop_sizes)
    t_duration = time_obj.step_size
    # initial time step is equal to initial population size
    times.append(time_obj.t[0])
    popsizes.append(copy.deepcopy(prev_popsizes))
    result = policy_obj.run(time_obj, env_obj)
    # start growing population at the second time step
    for time_ind, time_bin in enumerate(time_obj.t[1:]):
        # for each time bin, we get the posterior
        # probability of each nutrient state
        prob_nutr_states = result["prob_nutrient_states"][time_ind]
        assert np.isclose(sum(prob_nutr_states), 1.), \
          "sum of nutrient state probabilities must be 1"
        ## assign population to each nutrient type based on the probabilities
        ## given by the policy
        # first get nutrient states currently existing
        # in the population - note we use a copy of previous
        # population size list
        curr_popsizes = list(prev_popsizes)
        # total population size (regardless of nutrient state)
        total_popsize = sum(curr_popsizes)
        # assign total population to nutrient states in proportion to their
        # probability under the policy
        for nutr, nutr_prob in enumerate(prob_nutr_states):
            curr_popsizes[nutr] = int(round(total_popsize * nutr_prob))
        # calculate the growth rates for each population type
        # (based on what nutrient the population is tuned to, and what nutrient
        # the environment has)
        if len(init_pop_sizes) != len(prob_nutr_states):
            print "init pop sizes: ", init_pop_sizes
            print "prob. of nutrient states: ", prob_nutr_states
            raise Exception, "Probability vector over nutrient needs to have " \
                             "same size as initial population vector."
        for nutr, nutr_pop_size in enumerate(curr_popsizes):
            # here we enforce the constraints of the environment
            # on the policy: if the nutrient state matches the
            # environment, use that nutrient's growth rate to grow
            # the population. if it doesn't, use the mismatch growth
            # rate as given by the environment object.
            if nutr == env_obj.hist[time_ind]:
                nutr_growth_rate = env_obj.nutr_growth_rates[nutr]
                # in log space
                #curr_pop_size += ((t_duration * nutr_growth_rate) * np.log(2))
                # in non-logspace
                curr_popsizes[nutr] *= \
                  np.power(2, (t_duration * nutr_growth_rate))
            else:
                # population nutrient state doesn't match environment,
                # so use the mismatch growth rate
                curr_popsizes[nutr] *= \
                  np.power(2, (t_duration * env_obj.mismatch_growth_rate))
        # accumulate the current population size
        # for the time bin
        times.append(time_bin)
        # add a copy of curr_popsizes
        popsizes.append(list(curr_popsizes))
        # set previous population size to current population
        # size for next time point
        prev_popsizes = list(curr_popsizes)
    # check that we have population sizes for each time point
    assert (len(popsizes) == len(time_obj.t)), "Population sizes " \
           "do not match time interval."
    popsizes = np.array(popsizes)
    return popsizes

        
def simulate_individual_growth(init_pop_size, time_obj, growth_rates,
                               policy=None):
    """
    Simulate cell growth for each cell (individual) in a given population.
    This is a stochastic simulation. Return the population sizes
    in each interval as determined by time bins of 'time_obj'.

    Args:
    -----
    - init_pop_size: initial population size
    - time_obj: time interval over which to simulate
    - growth_rates: growth rates for each time interval (must be
      same size as time axis, time_obj.t)
    """
    assert (len(growth_rates) == len(time_obj.t)), \
           "Need growth rate for each time point."
    popsizes = []
    times = []
    # initial population size
    popsize = init_pop_size
    curr_t = time_obj.t_start
    num_timebins = len(time_obj.t)
    for time_ind, time_bin in enumerate(time_obj.t):
        growth_rate = growth_rates[time_ind]
        while curr_t < time_bin:
            # stochastically simulate cell divisons
            # here
            if growth_rate > 0:
                new_rate = 1/float(growth_rate * popsize)
                div_time = np.random.exponential(scale=new_rate)
                # advance time
                curr_t += div_time
                # increase population size
                popsize += 1
            else:
                # if the growth rate is zero for this time bin,
                # simply advance to next time bin
                curr_t = time_bin
        # accumulate the current population size
        # for the time bin
        times.append(time_bin)
        popsizes.append(popsize)
    # check that we have population sizes for each time point
    assert (len(popsizes) == len(time_obj.t)), "Population sizes " \
           "do not match time interval."
    return np.log2(np.array(popsizes))


def get_div_times_to_popsize(init_popsize, times):
    """
    Calculate the population size from time stamps of division
    times. Return population size at each time (i.e. return
    a numpy array of same length as 'times').

    Args:
    ----
    - init_popsize: initial population size
    - times: time stamps of division times. 
    """
    num_divs = len(times)
    divs_elapsed = np.cumsum(np.ones(num_divs))
    popsize = init_popsize * np.power(2, divs_elapsed)
    return popsize

def get_growth_rate(popsize_start, popsize_end, t_start, t_end):
    """
    Get growth rate per hour (doubling time).

    p_start * r * t * log2(2) = p_end

    r = p_end / (p_start * t * log2(2))
    """
    delta_t = t_end - t_start
    assert (delta_t > 0)
    r = popsize_end / (popsize_start * delta_t * np.log2(2))
    # make it growth rate per hour
    r = r / delta_t
    return r

class Growth:
    """
    Cell growth simulator.
    """
    def __init__(self, init_pop_sizes,
                 env_obj=None,
                 policy_obj=None,
                 params={}):
        """
        Args:
        -----
        - init_pop_sizes: list containing the number of cells in
        each nutrient state, e.g. [100, 0] to mean there
        are 100 cells in nutrient state 1, and 0 cells in nutrient state 2.
        The interpretation of nutrient number is given by the 'env_obj'.

        Kwargs:
        -------
        - env_obj: environment object
        - policy_obj: policy object
        - params: extra parameters
        """
        # initial population size (un-logged)
        self.init_pop_sizes = init_pop_sizes
        # current population size (un-logged)
        self.curr_pop_sizes = copy.deepcopy(self.init_pop_sizes)
        # environment to grow in
        self.env_obj = env_obj
        # growth policy object for cells
        self.policy_obj = policy_obj
        # related parameters
        self.params = params

    def simulate(self, time_obj, individual_growth=False):
        """
        Assume time interval starts at 0.

        Args:
        -----
        - time_obj: time interval over which to run simulation.

        Kwargs:
        -------
        - individual_growth: if True, simulate individual cells
          growth. [DEPRECATED]
        """
        if self.policy_obj is None:
            raise Exception, "Need growth policy to simulate growth."
        if self.env_obj is None:
            raise Exception, "Need environment to simulate growth."
        # apply policy to the environment. This determines the growth
        # rates at each time.
        policy_data = self.policy_obj.run(time_obj, self.env_obj)
        growth_rates = policy_data["growth_rates"]
        if not individual_growth:
            popsizes = \
              simulate_pop_growth(self.init_pop_sizes, time_obj,
                                  env_obj=self.env_obj,
                                  policy_obj=self.policy_obj)
        else:
            popsizes = \
              simulate_individual_growth(self.init_pop_sizes, time_obj,
                                         env_obj=self.env_obj,
                                         policy_obj=self.policy_obj)
        # population size in natural log
        log_popsizes = np.log(popsizes)
        growth_results = \
           {"t": time_obj,
            "growth_rates": growth_rates,
            "pop_sizes": popsizes,
            "log_pop_sizes": log_popsizes}
        # bring in other data from the policy
        growth_results.update(policy_data)
        return growth_results
            
    def simulate_constant_growth(self, growth_rate_per_hr, time_interval,
                                 time_num_bins=200):
        """
        Simulate constant growth.
        """
        growth_data = exp_growth(self.curr_pop_size,
                                 growth_rate_per_hr,
                                 time_interval,
                                 time_num_bins=time_num_bins)
        # Update current population size
        self.curr_pop_size = np.exp(growth_data["log_pop_size"][-1])
        return growth_data


def main():
    # stochastic division simulation
    # init_popsize = 1
    # time_obj = time_unit.Time(0, 10, step_size=1)
    # growth_rate = 1.
    # simulate_stoch_division(init_popsize, time_obj, growth_rate)
    pass
                 

if __name__ == "__main__":
    main()

