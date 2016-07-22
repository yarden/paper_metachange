##
## Growth policies.
##
## It up to the policy function to enforce the rules of growth, i.e.
## to 
##
##
import os
import sys
import time

import numpy as np
import pandas

import particlefever
import particlefever.particle_filter as particle_filter
import particlefever.switch_ssm as switch_ssm

import time_unit
import prob_utils

class GrowthPolicy:
    """
    Growth policy for a cell or population of cells.
    """
    def __init__(self, policy_func=None, policy_params={}):
        """
        - policy_func: function that as input an environment
          and a time axis, and returns various features.
        """
        self.policy_func = policy_func
        self.policy_params = policy_params
        self.result = {}

    def run(self, time_obj, env_obj):
        """
        Run policy on data.
        """
        ###
        ### Calculation of growth rate should be done here.
        ### Based on features of the environment.
        ###
        # run the policy on the environment to get the nutrient states
        policy_data = self.policy_func(time_obj, env_obj,
                                       params=self.policy_params)
        # calculate what the growth rate should be at each time point,
        # based on the policy's nutrient states and the environment
        growth_rates = []
        for t in range(len(time_obj.t)):
            env_state = env_obj.hist[t]
            cell_state = policy_data["nutrient_states"][t]
            # check if we have a match between environment
            # and cell state
            if env_state == cell_state:
                # environment is glucose and cell state is glucose
                rate = env_obj.nutr_growth_rates[env_state]
            else:
                # environment and cell state do not match
                rate = env_obj.mismatch_growth_rate
            growth_rates.append(rate)
        result = {"growth_rates": growth_rates,
                  "nutrient_states": policy_data["nutrient_states"]}
        # add additional data returned by policy into the result
        for policy_key in policy_data:
            if policy_key not in result:
                result[policy_key] = policy_data[policy_key]
        self.result = result
        return result


def make_switch_ssm_pf(params, num_particles=200):
    """
    Make particle filter object a switching state space model.
    """
    ###
    ### TODO: fix this so that the switch ssm model parameters
    ### from configuration file (*.params) are actually used here instead of
    ### the default switch ssm parameters
    ###
    ## instead of defining these parameters, rely on defaults of Switch SSM
    ## for these parameters
    num_switch_states = params["num_switch_states"]
    num_outputs = params["num_outputs"]
    # init_switch_hyperparams = np.array(params["init_switch_hyperparams"])
    # init_out_hyperparams = np.array(params["init_out_hyperparams"])
    # switch_trans_mat_hyperparams = np.array(params["switch_trans_mat_hyperparams"])
    # out_trans_mat_hyperparams = np.array(params["out_trans_mat_hyperparams"])
    # sticky_switch_weight = params["sticky_switch_weight"]
    ssm_pf = \
      particle_filter.DiscreteSwitchSSM_PF(num_switch_states,
                                           num_outputs,
                                           num_particles=num_particles)
    return ssm_pf



##
## growth policies without bet-hedging
##
def posterior_pred_policy(time_obj, env_obj, params):
    """
    Posterior predictive policy.
    """
    lag = params["decision_lag_time"]
    result = {"t": time_obj.t}
    data = env_obj.hist
    # determine growth rates based on filtering
    ssm_pf = make_switch_ssm_pf(params)
    ssm_pf.initialize()
    data = np.array(data)
    ssm_pf.process_data(data, save_steps=True)
    preds_with_lag = ssm_pf.prediction_with_lag(data, lag=lag)
    num_nutrients = len(params["nutr_labels"])
    num_timepoints = len(time_obj.t)
    nutrient_states = [np.argmax(pred) for pred in preds_with_lag]
    result["nutrient_states"] = nutrient_states
    result["prob_nutrient_states"] = np.zeros((num_timepoints,
                                               num_nutrients))
    # use maximum posterior strategy
    for n in xrange(len(time_obj.t)):
        max_prob_nutr = preds_with_lag[n, :].argmax()
        result["prob_nutrient_states"][n, max_prob_nutr] = 1.
    return result

def plastic_growth_policy(time_obj, env_obj, params):
    """
    The plastic growth policy from Jablonka (1995). Match whatever
    nutrient environment we saw in previous time step (based
    on what the lag in the environment is.)
    """
    ####
    #### TODO: incorporate the lag here
    ####
    result = {"t": time_obj.t}
    data = env_obj.hist
    # choose initial state randomly
    prev_data = prob_utils.sample_binary_state(0.5)
    result["nutrient_states"] = []
    num_nutrients = len(params["nutr_labels"])
    num_timepoints = len(time_obj.t)
    result["prob_nutrient_states"] = np.zeros((num_timepoints,
                                               num_nutrients))
    for n in xrange(num_timepoints):
        result["nutrient_states"].append(prev_data)
        result["prob_nutrient_states"][n, prev_data] = 1
        prev_data = data[n]
    return result

def glu_only_growth_policy(time_obj, env_obj, params):
    """
    Glucose-only growth policy. Independent of lag.
    """
    result = {"t": time_obj.t}
    data = env_obj.hist
    # choose initial state randomly
    result["nutrient_states"] = []
    num_nutrients = len(params["nutr_labels"])
    num_timepoints = len(time_obj.t)
    glu_val = env_obj.nutr_labels.index("glucose")
    result["prob_nutrient_states"] = np.zeros((num_timepoints,
                                               num_nutrients))
    for n in xrange(len(time_obj.t)):
        result["nutrient_states"].append(glu_val)
        result["prob_nutrient_states"][n, glu_val] = 1.
    return result

def rand_growth_policy(time_obj, env_obj, params):
    """
    Random growth policy. Independent of lag.
    """
    num_nutrs = len(params["nutr_labels"])
    num_timepoints = len(time_obj.t)
    result = {"t": time_obj.t,
              "prob_nutrient_states": np.zeros((num_timepoints,
                                                num_nutrs)),
              "nutrient_states": []}
    # random probability of nutrient
    probs = np.ones(num_nutrs) / float(num_nutrs)
    for t in xrange(len(time_obj.t)):
        nutr_val = np.random.multinomial(1, probs).argmax()
        # choose one nutrient randomly, without bet-hedging
        result["prob_nutrient_states"][t, nutr_val] = 1
        result["nutrient_states"].append(nutr_val)
    return result


def true_growth_policy(time_obj, env, params):
    # use the true probability parameters
    true_gluc_to_gluc = params["true_gluc_to_gluc"]
    true_galac_to_gluc = params["true_galac_to_gluc"]
    gluc_val = params["nutr_labels"].index("glucose")
    galac_val = params["nutr_labels"].index("galactose")
    no_growth_rate = params["mismatch_growth_rate"]
    prior_carbon_state = params["prior_carbon_state"]
    # initial prediction based on model
    result = {"nutrient_states": [],
              "growth_rates": [],
              "prob_nutrient_states": np.zeros((time_obj.num_steps, 2))}
    prev_state = env.hist[0]
    decision_lag_time = params["decision_lag_time"]
    result["nutrient_states"] = np.array([False] * time_obj.num_steps)
    # choose first state randomly
    nutrient_state = prob_utils.flip(prior_carbon_state)
    n = 0
    for time_int in time_obj.iter_interval_ind(decision_lag_time):
        curr_interval, (start, end) = time_int
        obs = env.hist[start]
        # determine intervening sequences
        result["nutrient_states"][start:end] = nutrient_state
        # take max probability action depending on previous state
        if obs == gluc_val:
            # current observation is glucose
            cond_prob = true_gluc_to_gluc
        else:
            # current observation is galactose
            cond_prob = true_galac_to_gluc
        if cond_prob >= 0.5:
            nutrient_state = gluc_val
        else:
            nutrient_state = galac_val
        result["prob_nutrient_states"][n, nutrient_state] = 1.
        n += 1
    result["t"] = time_obj.t
    return result


##
## bet-hedging growth policies
##
def bh_particle_filter_policy(time_obj, env_obj, 
                              growth_obj=None,
                              lag=1,
                              **model_params):
    # make particle filter for inference
    params = model_params["params"]
    lag = params["decision_lag_time"]
    pf_obj = make_switch_ssm_pf(params)
    pf_obj.initialize()
    data = np.array(env_obj.hist)
    pf_obj.process_data(data)
    pred_posteriors = pf_obj.prediction_with_lag(data, lag=lag)
    # choose nutrient state according to maximum
    # posterior probability
    nutrient_states = [np.argmax(p) for p in pred_posteriors]
    results = {"nutrient_states": nutrient_states,
               "prob_nutrient_states": pred_posteriors}
    return results
    

def bh_rand_growth_policy(time_obj, env_obj,
                          growth_obj=None,
                          params={}):
    """
    Cell growth policy where cells randomly
    switch between glucose and galactose nutrient states
    but grow faster on glucose.
    """
    num_nutrs = len(params["nutr_labels"])
    # random probability of nutrient state (with bet-hedging)
    num_timepoints = len(time_obj.t)
    prob_nutrient_states = \
      np.ones((num_timepoints, num_nutrs)) / float(num_nutrs)
    result = {"prob_nutrient_states": prob_nutrient_states,
              "nutrient_states": []}
    nutrient_states = []
    for t in time_obj.t:
        nutr_val = \
          np.random.multinomial(1, prob_nutrient_states[t, :]).argmax()
        result["nutrient_states"].append(nutr_val)
    return result

def bh_glu_only_growth_policy(time_obj, env_obj,
                              growth_obj=None,
                              params={}):
    """
    Cell growth policy where cells randomly
    switch between glucose and galactose nutrient states
    but grow faster on glucose.
    """
    glu_growth_rate = 0.3
    gal_growth_rate = glu_growth_rate / 7.
    # numeric value associated with glucose/galactose
    glu_val = 0
    gal_val = 1
    nutrient_states = []
    for t in time_obj.t:
        nutrient_states.append(glu_val)
    result = {"nutrient_states": nutrient_states}
    return result





