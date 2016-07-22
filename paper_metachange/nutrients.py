##
##
##
import os
import sys
import time

import matplotlib
import matplotlib.pylab as plt

import numpy as np
import pandas

import seaborn as sns
import utils

sns.set_style("ticks")

import time_unit
import prob_utils
import paths

import peripy
import peripy.flow_simulator as flow_sim
from collections import OrderedDict

##
## Continuous nutrient level simulators
##
def noisy_cont_glucose(time_axis, gluc_mean=1, gluc_sd=0.08, min_value=0):
    """
    Noisy continuous glucose levels. Glucose levels vary as Gaussian
    random walk.
    """
    num_timepoints = len(time_axis)
    data = []
    gluc_val = np.random.normal(loc=gluc_mean, scale=gluc_sd)
    # initial glucose value
    data.append(gluc_val)
    # sample remaining glucose values
    for n in range(num_timepoints - 1):
        next_gluc_val = max(np.random.normal(loc=gluc_val, scale=gluc_sd),
                            min_value)
        data.append(next_gluc_val)
        gluc_val = next_gluc_val
    return data


def noisy_glucose_flow(time_obj, initial_amount, volume, times_to_flows,
                       gluc_mean=0,
                       gluc_sd=0.05,
                       min_value=0,
                       time_bin=0.1):
    """
    Noisy glucose flow based on ODEs.
    """
    flow_obj = flow_sim.FlowSim(volume)
    sim_results = flow_obj.simulate_flow(initial_amount, times_to_flows,
                                         time_bin=time_bin)
    t = sim_results["times"]
    # add Gaussian noise to results
    noise = np.random.normal(loc=gluc_mean, scale=gluc_sd, size=len(t))
    sim_results["amounts"] += noise
    return sim_results


def semimarkov_switch_glucose(time_obj, initial_amount, volume, durations,
                              params):
    """
    Semi-Markov switching glucose model.

    Pulse durations are rate parameters for Poisson distribution.

    - durations: list of lambda parameters for Poisson-distributed
      durations for [no_pulse, pulse]
    """
    pulse_gluc_mu = params["pulse_gluc_mu"]
    pulse_gluc_sd = params["pulse_gluc_sd"]
    no_pulse_gluc_mu = params["no_pulse_gluc_mu"]
    no_pulse_gluc_sd = params["no_pulse_gluc_sd"]
    obs_sd = params["obs_sd"]
    prior_pulse = params["prior_pulse"]
    p_pulse_to_pulse = params["p_pulse_to_pulse"]
    p_no_pulse_to_pulse = params["p_no_pulse_to_pulse"]
    # parameters that are governed by switch state
    means = [no_pulse_gluc_mu, pulse_gluc_mu]
    sds = [no_pulse_gluc_sd, pulse_gluc_sd]
    def get_pulse(prev_pulse=None):
        if prev_pulse is None:
            # draw from prior
            pulse = (np.random.rand() <= prior_pulse)
        else:
            # draw based on previous pulse (governed by
            # pulse transition probabilities)
            pulse = prob_utils.sample_trans_prob(prev_pulse,
                                                 p_pulse_to_pulse,
                                                 p_no_pulse_to_pulse)
        return pulse
    def get_amounts(pulse=None, prev_amount=None, durations=[]):
        # draw duration depending on whether there is
        # a pulse or not
        lam = durations[int(pulse)]
        duration = np.random.poisson(lam=lam)
        amounts = []
        if prev_amount is None:
            # prior amount for first time point
            prev_amount = initial_amount
        for n in range(duration):
            switch_ind = int(pulse)
            to_add = np.random.normal(loc=means[switch_ind],
                                      scale=sds[switch_ind])
            amount = max(prev_amount + to_add, 0)
            amounts.append(amount)
        return amounts, duration
    def get_obs(amounts):
        obs = [max(amount + np.random.normal(loc=0, scale=obs_sd), 0) \
               for amount in amounts]
        return obs
    # initialize states
    pulses = [get_pulse()]
    amounts, curr_duration = get_amounts(pulse=pulses[0], durations=durations)
    all_durations = [curr_duration]
    obs = [get_obs(amounts)]
    num_steps = len(time_obj.t)
    for t in time_obj.t[1:]:
        curr_pulse = get_pulse(pulses[-1])
        curr_amounts, curr_duration = \
          get_amounts(pulse=curr_pulse,
                      prev_amount=amounts[-1],
                      durations=durations)
        curr_obs = get_obs(curr_amounts)
        all_durations.append(curr_duration)
        pulses.append(curr_pulse)
        amounts.extend(curr_amounts)
        obs.extend(curr_obs)
    # truncate results to be within boundary of time
    # interval
    results = {"pulses": pulses[0:num_steps],
               "amounts": np.array(amounts[0:num_steps]),
               "obs": obs[0:num_steps],
               "durations": all_durations,
               "times": time_obj.t}
    return results


def linear_switch_glucose(time_obj, initial_amount, volume,
                          params):
    """
    Switching linear state space model for glucose levels.

    Markov process for pulses:

      P_t | P_t-1 ~ Markov(p_pulse_to_pulse, p_no_pulse_to_pulse)

    """
    pulse_gluc_mu = params["pulse_gluc_mu"]
    pulse_gluc_sd = params["pulse_gluc_sd"]
    no_pulse_gluc_mu = params["no_pulse_gluc_mu"]
    no_pulse_gluc_sd = params["no_pulse_gluc_sd"]
    obs_sd = params["obs_sd"]
    prior_pulse = params["prior_pulse"]
    p_pulse_to_pulse = params["p_pulse_to_pulse"]
    p_no_pulse_to_pulse = params["p_no_pulse_to_pulse"]
    # parameters that are governed by switch state
    means = [no_pulse_gluc_mu, pulse_gluc_mu]
    sds = [no_pulse_gluc_sd, pulse_gluc_sd]
    def get_pulse(prev_pulse=None):
        if prev_pulse is None:
            # draw from prior
            pulse = (np.random.rand() <= prior_pulse)
        else:
            # draw based on previous pulse (governed by
            # pulse transition probabilities)
            pulse = prob_utils.sample_trans_prob(prev_pulse,
                                                 p_pulse_to_pulse,
                                                 p_no_pulse_to_pulse)
        return pulse
    def get_amount(pulse=None, prev_amount=None):
        if pulse is None:
            # draw from prior
            amount = initial_amount
        else:
            assert(prev_amount != None), "prev amount must be given."
            switch_ind = int(pulse)
            to_add = np.random.normal(loc=means[switch_ind],
                                      scale=sds[switch_ind])
            amount = prev_amount + to_add
        return max(amount, 0)
    def get_obs(amount):
        obs = amount + np.random.normal(loc=0, scale=obs_sd)
        return max(obs, 0)
    # initialize states
    pulses = [get_pulse()]
    amounts = [get_amount(pulse=pulses[0],
                          prev_amount=initial_amount)]
    obs = [get_obs(amounts[0])]
    num_steps = len(time_obj.t)
    for t in time_obj.t[1:]:
        curr_pulse = get_pulse(pulses[-1])
        curr_amount = get_amount(pulse=curr_pulse,
                                 prev_amount=amounts[-1])
        curr_obs = get_obs(curr_amount)
        pulses.append(curr_pulse)
        amounts.append(curr_amount)
        obs.append(curr_obs)
    results = {"pulses": pulses,
               "amounts": np.array(amounts),
               "obs": obs,
               "times": time_obj.t}
    return results



##
## Rate-based models
##
def semimarkov_rate_glucose(time_obj, initial_amount, volume, durations,
                            params):
    """
    Semi-Markov glucose rate-based model.

    1. Choose whether or not we have a pulse

    2. Choose rate R_t given pulse

    3. Choose duration d given pulse

    4. Generate sequence of observations
    """
    pulse_rate_mu = params["pulse_rate_mu"]
    pulse_rate_sd = params["pulse_rate_sd"]
    no_pulse_rate_mu = params["no_pulse_rate_mu"]
    no_pulse_rate_sd = params["no_pulse_rate_sd"]
    obs_sd = params["rate_obs_sd"]
    prior_pulse = params["prior_pulse"]
    p_pulse_to_pulse = params["p_pulse_to_pulse"]
    p_no_pulse_to_pulse = params["p_no_pulse_to_pulse"]
    # parameters that are governed by switch state
    means = [no_pulse_rate_mu, pulse_rate_mu]
    sds = [no_pulse_rate_sd, pulse_rate_sd]
    def get_pulse(prev_pulse=None):
        if prev_pulse is None:
            # draw from prior
            pulse = (np.random.rand() <= prior_pulse)
        else:
            # draw based on previous pulse (governed by
            # pulse transition probabilities)
            pulse = prob_utils.sample_trans_prob(prev_pulse,
                                                 p_pulse_to_pulse,
                                                 p_no_pulse_to_pulse)
        return pulse
    def get_amounts(pulse=None, prev_amount=None, durations=[]):
        # draw duration depending on whether there is
        # a pulse or not
        lam = durations[int(pulse)]
        duration = np.random.poisson(lam=lam)
        amounts = []
        if prev_amount is None:
            # prior amount for first time point
            prev_amount = initial_amount
        # sample the rate for current duration
        switch_ind = int(pulse)
        curr_rate = np.random.normal(loc=means[switch_ind],
                                     scale=sds[switch_ind])
        for n in range(duration):
            # produce observations for the duration
            # conditioned on the rate
            # G(t) = G(t) + G(t)*R(t)*delta_t
            latent_amount = prev_amount + \
                            (prev_amount * curr_rate * time_obj.step_size)
            curr_rate = np.random.normal(loc=means[switch_ind],
                                         scale=sds[switch_ind])
            # observation noise on amount
            to_add = np.random.normal(0, obs_sd)
            amount = max(latent_amount + to_add, 0)
            amounts.append(amount)
            # advance latent amount
            prev_amount = latent_amount
        return amounts, duration
    def get_obs(amounts):
        obs = [max(amount + np.random.normal(loc=0, scale=obs_sd), 0) \
               for amount in amounts]
        return obs
    # initialize states
    pulses = [get_pulse()]
    amounts, curr_duration = get_amounts(pulse=pulses[0], durations=durations)
    all_durations = [curr_duration]
    obs = [get_obs(amounts)]
    num_steps = len(time_obj.t)
    for t in time_obj.t[1:]:
        curr_pulse = get_pulse(pulses[-1])
        curr_amounts, curr_duration = \
          get_amounts(pulse=curr_pulse,
                      prev_amount=amounts[-1],
                      durations=durations)
        curr_obs = get_obs(curr_amounts)
        all_durations.append(curr_duration)
        pulses.append(curr_pulse)
        amounts.extend(curr_amounts)
        obs.extend(curr_obs)
    results = {"pulses": pulses[0:num_steps],
               "amounts": np.array(amounts[0:num_steps]),
               "obs": obs[0:num_steps],
               "durations": all_durations,
               "times": time_obj.t}
    return results


####
#### TODO
####
# 1. fit model where the parameters are rates, and you have to infer rate
#    from noisy observation and predict when the nutrient will run out
#
# 2. fit more sophisticated model where the parameters measured are abundances
#    and you have to compute a rate:
#
#       t2 - t1/time_bin (deterministic function of random variables.) 
#
#  - question: does this connect to the question of why ratios are useful?

##
## Discrete nutrient level simulators
##
def discrete_markov_glucose(time_obj,
                            switch_interval=1,
                            gluc_on_value=1,
                            gluc_off_value=0,
                            p_gluc=0.5,
                            p_gluc_given_on=0.9,
                            p_gluc_given_off=0.2):
    """
    Discrete Markov glucose levels.
    """
    def sample_next_gluc(prev_val):
        if prev_val == 1:
            next_gluc_val = np.random.binomial(1, p_gluc_given_on)
        else:
            next_gluc_val = np.random.binomial(1, p_gluc_given_off)
        return next_gluc_val
    # sample from prior glucose level
    gluc_val = np.random.binomial(1, p_gluc)
    data = []
    for curr_interval in time_obj.iter_interval(switch_interval):
        data.extend([gluc_val] * len(curr_interval))
        gluc_val = sample_next_gluc(gluc_val)
    # Replace with designated values for glucose on/glucose off
    data = utils.replace(data, 1, gluc_on_value)
    data = utils.replace(data, 0, gluc_off_value)
    return data


def discrete_markov_glucose_galactose(time_obj,
                                      true_gluc_to_gluc,
                                      true_galac_to_gluc,
                                      p_carbon_state=0.5,
                                      init_state=True):
    if init_state is None:
        init_state = (np.random.rand() <= p_carbon_state)
    data = [init_state]
    if len(time_obj.t) == 1:
        return data
    points = range(len(time_obj.t))
    for n in points[1:]:
        if data[n-1] == True:
            # probability of transitioning from gluc to gluc
            next_state = (np.random.rand() <= true_gluc_to_gluc)
        else:
            next_state = (np.random.rand() <= true_galac_to_gluc)
        data.append(next_state)
    return data


def discrete_determ_switch_glucose(time_obj,
                                   gluc_on_value=0.8,
                                   gluc_off_value=0,
                                   switch_interval=1):
    """
    Discrete deterministic switch of glucose on/off.

    Switches deterministically every 'switch_interval'-length
    interval between glucose ON and glucose OFF.
    """
    data = []
    # start with off
    gluc_val = gluc_off_value
    for curr_interval in time_obj.iter_interval(switch_interval):
        next_values = [gluc_val] * len(curr_interval)
        data.extend(next_values)
        # reverse glucose values
        if gluc_val == gluc_on_value:
            gluc_val = gluc_off_value
        else:
            gluc_val = gluc_on_value
    return data


def plot_nutrient_decays():
    """
    Plot different nutrient decay dynamics.
    """
    time_obj = time_unit.Time(0, 60, 0.1)
#    time_obj = time_unit.Time(0, 100, 10)
    ### Decay regimen
    flow_rates = [1.5, 5, 12, 20]
    flow_end_times = [20, 20, 20, 20]
    initial_amounts = [0.75, 1.0, 2., 4.5]
    equilibrium_rate = 1.
#    flow_rates = [5, 10]
    volume = 200
    dfs = []
    for n in xrange(len(flow_rates)):
        rate = flow_rates[n]
        flow_end = flow_end_times[n]
        initial_amount = initial_amounts[n]
        df = {}
        times_to_flows = OrderedDict()
        times_to_flows[0] = {"flow_in": rate,
                             "flow_out": rate,
                             "flow_in_conc": 0.}
        times_to_flows[flow_end] = {"flow_in": rate,
                                    "flow_out": rate,
                                    "flow_in_conc": 0.}
        assert (flow_end <= time_obj.t_end), \
          "Flow end must be less than %.2f" %(float(time_obj.t_end))
        if flow_end < time_obj.t_end:
            # rest of the time do equilibrium
            equilibrium_flow = {"flow_in": 0.,
                                "flow_out": 0.,
                                "flow_in_conc": 0.}
            times_to_flows[flow_end + time_obj.step_size] = \
              equilibrium_flow
            times_to_flows[time_obj.t_end] = equilibrium_flow
        sim_results = \
          noisy_glucose_flow(time_obj, initial_amount, volume, times_to_flows,
                             time_bin=time_obj.step_size,
                             gluc_sd=0.001)
        print "sim results: ", len(sim_results["amounts"]), len(sim_results["times"])
        df["rate"] = rate
        df["Amounts"] = sim_results["amounts"]
        df["Time"] = sim_results["times"]
        df = pandas.DataFrame(df)
        dfs.append(df)
    merged_df = pandas.concat(dfs, ignore_index=True)
    merged_df = merged_df.rename(columns={"rate": "Flow rate (ml/min)"})
#    sim_df = pandas.melt(df, id_vars=["Time"]).rename(columns={"variable": "Flow rate (ml/min)"})
    sns.set_style("ticks")
    sns.tsplot(time="Time", value="Amounts", unit="Flow rate (ml/min)",
               condition="Flow rate (ml/min)",
               linewidth=3,
               color=sns.color_palette("Blues"),
#               color=sns.color_palette("Greys")[::-1][0:len(flow_rates)][::-1],
               data=merged_df)
    plt.xlabel("Time (min)")
    plt.ylabel("Glucose (mM)")
    plt.title(r"Initial amount: $%.1f$ mM, $V = %.1f$ ml" %(float(initial_amount),
                                                  float(volume)))
    plt.xlim([time_obj.t.min(), time_obj.t.max()])
    plt.ylim([0, max(initial_amounts)])
    sns.despine(trim=True, offset=2)


def plot_nutrient_pulses():
    step_size = 0.1
    time_obj = time_unit.Time(0, 100, step_size)
    ### Decay regimen
    initial_amount = 0
    flow_rates = [1, 5, 10, 20]
    volume = 200
    df = {}
    flow_in_conc = 0.1
    # interval for flow in 
    flow_in_interval = [0, 10]
    # interval for flow out
    flow_out_interval = [10 + step_size, 100]
    for rate in flow_rates:
        times_to_flows = OrderedDict()
        for fi in flow_in_interval:
            times_to_flows[fi] = {"flow_in": rate,
                                 "flow_out": rate,
                                 "flow_in_conc": flow_in_conc}
            times_to_flows[fi] = {"flow_in": rate,
                                  "flow_out": rate,
                                  "flow_in_conc": flow_in_conc}
        for fo in flow_out_interval:
            times_to_flows[fo] = {"flow_in": rate,
                                  "flow_out": rate,
                                  "flow_in_conc": 0.}
        sim_results = \
          noisy_glucose_flow(time_obj,
                             initial_amount,
                             volume,
                             times_to_flows,
                             time_bin=time_obj.step_size)
        df[rate] = sim_results["amounts"]
        df["Time"] = sim_results["times"]
    df = pandas.DataFrame(df)
    sim_df = \
      pandas.melt(df, id_vars=["Time"]).rename(columns={"variable":
                                                        "Flow rate (ml/min)"})
    plt.figure()
    sns.tsplot(time="Time", value="value", unit="Flow rate (ml/min)",
               condition="Flow rate (ml/min)",
#               color=sns.color_palette("Blues"),
               color=sns.color_palette("Greys")[::-1][0:len(flow_rates)][::-1],
               data=sim_df)
    plt.xlabel("Time (min)")
    plt.ylabel("Glucose (mM)")
    plt.title(r"Initial amount: $%.1f$ mM, $V = %.1f$ ml, $c_{in}$ = %.1f mM" \
              %(float(initial_amount),
                float(volume),
                float(flow_in_conc)))
    max_amount = sim_df["value"].max()
    plt.ylim([0, round(max_amount + 1)])
    y_coord = 0.5
    plt.hlines(y_coord, flow_in_interval[0], flow_in_interval[1],
               colors="r", zorder=10, linewidth=4)
    plt.hlines(y_coord, flow_out_interval[0], flow_out_interval[1],
               colors="darkblue", zorder=10, linewidth=4)
    plot_fname = os.path.join(paths.PLOTS_DIR, "nutrient_pulses_rates.pdf")
    sns.despine()
    plt.savefig(plot_fname)
    plt.show()


def plot_cont_nutrient_pulses():
    def plot_pulses(results, ymin=0, ymax=20):
        plt.plot(results["times"], results["amounts"])
        s = np.array([1] * len(results["times"]))
        c = np.array(["k"] * len(results["times"]))
        if "durations" in results:
            # semi-Markovian
            start = 0
            for d, pulse in zip(results["durations"],
                                results["pulses"]):
                end = min(start + d, len(results["times"]) - 1)
                if pulse:
                    c[start:end] = "red"
                    s[start:end] = 2
                start += d
        else:
            # Markovian
            for n, t in enumerate(results["times"]):
                pulse = results["pulses"][n]
                if pulse:
                    c[n] = "red"
                    s[n] = 2
        plt.scatter(results["times"], [1] * len(results["times"]), color=c, s=s)
        plt.xlabel(r"Time, $t$")
        plt.ylabel("Glucose amount")
        plt.ylim([ymin, ymax])
        plt.xlim([time_obj.t.min(), time_obj.t.max()])
        sns.despine()
#    step_size = 1/60.
#    time_obj = time_unit.Time(0, 10, step_size)
    step_size = 1/10.
    time_obj = time_unit.Time(0, 100, step_size)
    initial_amount = 3
    volume = 200
    #results = cont_linear_markov_glucose_pulses(time_obj, initial_amount, volume)
    #results = cont_rate_markov_glucose(time_obj, initial_amount, volume)
    # Compare Markov results to semi-Markov results
    params = {"pulse_gluc_mu": 1.5,
              "pulse_gluc_sd": 0.1,
              "no_pulse_gluc_mu": -1.0,
              "no_pulse_gluc_sd": 0.1,
              "obs_sd": 0.5,
              "prior_pulse": 0.001,
              "p_pulse_to_pulse": 0.8,
              "p_no_pulse_to_pulse": 0.1,
              "rate_obs_sd": 0.1,
              "pulse_rate_mu": 0.6,
              "pulse_rate_sd": 0.001,
              "no_pulse_rate_mu": -0.2,
              "no_pulse_rate_sd": 0.001}
#    np.random.seed(2)
    markov_results = linear_switch_glucose(time_obj, initial_amount, volume,
                                           params)
    durations = [10, 3]
    semimarkov_results = semimarkov_switch_glucose(time_obj, initial_amount, volume,
                                                   durations,
                                                   params)
    semimarkov_rate_results = \
      semimarkov_rate_glucose(time_obj, initial_amount, volume, durations,
                              params)
    ymin = 0
    ymax = max([markov_results["amounts"].max(),
                semimarkov_results["amounts"].max(),
                semimarkov_rate_results["amounts"].max()])
    ymax=None
    plt.figure()
    plt.subplot(3, 1, 1)
    plot_pulses(markov_results, ymin=ymin, ymax=ymax)
    plt.title("Markov glucose pulses (step size = %.3f)" %(step_size))
    plt.subplot(3, 1, 2)
    plot_pulses(semimarkov_results, ymin=ymin, ymax=ymax)
    plt.title("semi-Markov glucose pulses (step size = %.3f), durations = [%d, %d]" \
              %(step_size, durations[0], durations[1]))
    plt.subplot(3, 1, 3)
    plot_pulses(semimarkov_rate_results, ymin=ymin, ymax=ymax)
    plt.title("semi-Markov rate pulses (step size = %.3f), durations = [%d, %d]" \
              %(step_size, durations[0], durations[1]))
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    plot_nutrient_decays()
    plt.show()
#    plot_cont_nutrient_pulses()
#    plot_nutrient_pulses()


    #data = discrete_determ_switch_glucose(time_obj)
    #for x in time_obj.iter_interval(2):
    #    print x
#    data = discrete_markov_glucose(time_obj)
#    print len(data)
#    print len(time_obj.t)
#    plt.plot(time_obj.t, data, '-o')
#    plt.show()
    ##
    ## test out flow params
    ##
    time_obj = time_unit.Time(0, 10, 1/60.)
    times_to_flows = OrderedDict()
    ### Decay regimen
    initial_amount = 2
    times_to_flows[0] = {"flow_in": 20,
                         "flow_out": 20,
                         "flow_in_conc": 0.}
    times_to_flows[100] = {"flow_in": 30,
                           "flow_out": 30,
                           "flow_in_conc": 0.}
    ### Pulse regimen
    #initial_amount = 0
    # # t = 0: zero flows
    # times_to_flows[0] = {"flow_in": 0,
    #                      "flow_out": 0,
    #                      "flow_in_rpm": 0,
    #                      "flow_out_rpm": 0,
    #                      "flow_in_conc": 0.}
    # # t = 5: flow in 50 g per liter 
    # times_to_flows[5] = {"flow_in": 30,
    #                      "flow_out": 30,
    #                      "flow_in_conc": 50/1000.}
    # # t = 10: flow in clean water
    # times_to_flows[10] = {"flow_in": 30,
    #                       "flow_out": 30,
    #                       "flow_in_conc": 0.}
    # times_to_flows[50] = {"flow_in": 30,
    #                       "flow_out": 30,
    #                       "flow_in_conc": 0.}
    volume = 200
    sim_results = \
      noisy_glucose_flow(time_obj, initial_amount, volume, times_to_flows,
                         time_bin=time_obj.step_size)
    # plt.figure()
    # plt.plot(sim_results["times"], sim_results["amounts"])
    # plt.xlabel("Time (min)")
    # plt.ylabel("Glucose (mM)")
    # plt.ylim([0, 2.5])
    # sns.despine()
    # plt.show()
    
    
    
