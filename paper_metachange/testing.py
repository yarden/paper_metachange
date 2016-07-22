##
## Unit testing for code
##
import os
import sys
import time
import unittest

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

import pandas

import time_unit
import prob_utils
import nutrients
import fitness
import policies
import growth
import env
import nutrient_simulators

from collections import OrderedDict

class TestTime(unittest.TestCase):
    """
    Test time unit representation.
    """
    def test_time_length(self):
        # check time unit lengths
        print "testing time unit length"
        time_obj = time_unit.Time(1, 10, step_size=1)
        assert (len(time_obj.t) == 10), "Expected 10 steps in [1, 10]."

    def test_interval_to_bins(self):
        # check conversion from time interval to number of bins
        print "test interval to bins"
        # [1, 10] with step 1 means: 1 time unit takes up 1 bin
        time_obj = time_unit.Time(1, 10, step_size=1)
        assert (time_obj.get_num_bins_to(1) == 1)
        assert (time_obj.get_num_bins_to(2) == 2)
        assert (time_obj.get_num_bins_to(10) == 10)
        assert (time_obj.get_num_bins_to(1.5) == 2)
        assert (time_obj.get_num_bins_to(2.5) == 3)        
        # [1, 10] with step 0.5 means: 1 time unit takes up 2 bins
        time_obj = time_unit.Time(1, 10, step_size=0.5)
        assert (time_obj.get_num_bins_to(1) == 2)
        assert (time_obj.get_num_bins_to(2) == 4)
        assert (time_obj.get_num_bins_to(1.5) == 3)

    def test_inter_interval(self):
        def get_results_from_env(env, time_obj, lag):
            default_val = -900
            last_val = -1
            results = np.array([default_val] * len(time_obj.t))
            for curr_int in time_obj.iter_interval_ind(lag):
                interval, (start, end) = curr_int
                results[start:end] = last_val
                obs = env[start]
                last_val = int(obs)
            return results
        # test iteration by intervals
        print "testing iteration by intervals"
        env = np.array([True, False, True, False])
        time_obj = time_unit.Time(1, 4, step_size=1)
        assert (time_obj.num_steps == 4)
        # test even size lag: lag = 1
        lag = 1
        results = get_results_from_env(env, time_obj, lag)
        # initial value
        assert (np.array_equal(results, np.array([-1, 1, 0, 1])) == True)
        # test lag = 2
        lag = 2
        results = get_results_from_env(env, time_obj, lag)
        assert (np.array_equal(results, np.array([-1, -1, 1, 1])) == True)
        # test lag = 3
        lag = 3
        results = get_results_from_env(env, time_obj, lag)
        assert (np.array_equal(results, np.array([-1, -1, -1, 1])) == True)
        

class TestProbUtils(unittest.TestCase):
    """
    Test probability functions.
    """
    # periodic transition matrix
    trans_mat = np.array([[0., 1.],
                          [1., 0.]])
    # initial probabilities concentrated on first state
    init_probs = np.array([1., 0.])
    num_samples = 10
    samples1 = prob_utils.sample_markov_chain(num_samples,
                                              init_probs,
                                              trans_mat)
    print "expecting periodic samples: ", samples1
    # make sure we get periodic samples
    assert (samples1 == np.array([0, 1] * 5)).all(), \
      "Expected periodic samples."
    trans_mat = np.array([[1., 0.],
                          [1., 0.]])
    samples2 = prob_utils.sample_markov_chain(num_samples,
                                              init_probs,
                                              trans_mat)
    # make sure we get constant samples
    print "expecting constant samples: ", samples2
    assert (samples2 == np.array([0, 0] * 5)).all(), \
      "Expected periodic samples."


class TestGrowth(unittest.TestCase):
    """
    Testing growth functions.
    """
    def test_growth_popsize(self):
        # check that we get the right population size
        init_popsize = 1
        # after 10 time steps, population size should be 2**10
        # regardless of step sizes
        step_sizes = [1, 0.1, 0.5]
        for step_size in step_sizes:
            print "testing growth with step size: %.2f" %(step_size)
            t_start = 0
            t_end = 10
            growth_duration = t_end - t_start
            time_obj = time_unit.Time(t_start, t_end, step_size=step_size)
            num_steps = len(time_obj.t)
            growth_rates = np.array([1.] * num_steps)
            log_pop_size = growth.simulate_growth(init_popsize,
                                                  time_obj,
                                                  growth_rates)
            final_count = np.exp(log_pop_size[-1])
            expected_final_count = np.power(2, growth_duration)
            assert (np.allclose(final_count, expected_final_count)), \
                   "Expected 2^10 cells at end of growth."


    def test_div_times_to_popsize(self):
        print "testing division times to population size calculation"
        # time stamp data for when divisions occurred
        div_times = [0.5, 1, 2, 3, 4.5]
        # population size = 1
        init_popsize = 1
        popsize = growth.get_div_times_to_popsize(init_popsize, div_times)
        assert (popsize[-1] == 2**5), "Expected 2^5 cells from division times."
        # population size = 5
        init_popsize = 5
        popsize = growth.get_div_times_to_popsize(init_popsize, div_times)
        assert (popsize[-1] == (init_popsize * 2**5)), \
          "Expected 5*(2^5) cells from division times."


    def _test_individual_growth(self):
        dfs = []
        time_obj = time_unit.Time(0, 10, step_size=1)
        print len(time_obj.t)
        # simulate environment
        def constant_env(time_obj):
            return [0] * len(time_obj.t)
        num_iters = 200
        print "simulating individual cell growth"
        init_pop_size = 1
        for n_iter in xrange(num_iters):
            # make on the fly a policy with the given constant
            # growth rate
            def constant_policy(time_obj, env, params={}):
                return {"nutrient_states": [0] * len(time_obj.t)}
            policy_obj = policies.GrowthPolicy(constant_policy)
            nutr_labels = ["glucose", "galactose"]
            nutr_simulator = constant_env
            env_obj = env.MixedDiscEnvironment(nutr_labels, nutr_simulator,
                                               mismatch_growth_rate=0.0,
                                               nutr_growth_rates=[0.3, 0.3/4])
            env_obj.simulate(time_obj)
            growth_obj = growth.Growth(init_pop_size,
                                       env=env_obj,
                                       policy=policy_obj)
            results = growth_obj.simulate(time_obj, individual_growth=True)
            df = pandas.DataFrame({"time": time_obj.t,
                                   "iter": n_iter,
                                   "pop_size":
                                   np.power(2, results["log_pop_size"]),
                                   "log_pop_size":
                                   results["log_pop_size"]})
            dfs.append(df)
        merged_df = pandas.concat(dfs, ignore_index=True)
        plt.figure()
        sns.set_style("ticks")
        sns.tsplot(time="time",
                   unit="iter",
                   err_style="unit_traces",
                   value="pop_size",
                   data=merged_df)
        plt.xlabel("Time step")
        plt.ylabel("Pop size")
        sns.despine(trim=True, offset=1)
        plot_fname = "./test_individual_growth.pdf"
        print "mean final population size: "
        print merged_df[merged_df["time"] == time_obj.t[-1]]["pop_size"].mean()
        print "Saving test results to: %s" %(plot_fname)
        plt.savefig(plot_fname)

    def _test_simple_bet_hedge_sim(self):
        """
        Test simple bet hedge simulation.
        """
        nutr_growth_rates = [1., 0.3]
        mismatch_growth_rate = 0.
        nutr_labels = ["glucose", "galactose"]
        ## test all glucose environment
        def all_glu_simulator(time_obj):
            glu_val = nutr_labels.index("glucose")
            return np.array([glu_val] * len(time_obj.t))
        nutr_simulator = all_glu_simulator
        # simulate environment
        env_obj = \
          env.MixedDiscEnvironment(nutr_labels, nutr_simulator,
                                   nutr_growth_rates=nutr_growth_rates,
                                   mismatch_growth_rate=mismatch_growth_rate)
        time_obj = time_unit.Time(0, 10, 1)
        num_timesteps = len(time_obj.t)
        # simulate the environment
        env_obj.simulate(time_obj)
        # use a posterior predictive bet hedging policy
        # based on a particle filter
        params = {"num_switch_states": 2,
                  "num_outputs": len(nutr_labels),
                  "decision_lag_time": 1,
                  "nutr_labels": nutr_labels}
        policy_obj = \
          policies.GrowthPolicy(policy_func=policies.bh_particle_filter_policy,
                                policy_params=params)
        # initial population sizes
        init_pop_sizes = [100, 0]
        growth_obj = growth.Growth(init_pop_sizes,
                                   env_obj=env_obj,
                                   policy_obj=policy_obj)
        results = growth_obj.simulate(time_obj)
        # first time bin population size should be equal to initial
        # population we started with
        popsizes = results["pop_sizes"]
        for n, p in enumerate(popsizes[0, :]):
            assert p == init_pop_sizes[n], "First population size " \
              "should equal initial population we started with."
        ##
        ## comparison of posterior predictive
        ## policy and bet-hedging
        ##
        ## test alternating environment
        time_obj = time_unit.Time(0, 9, 1)
        def glu_gal_simulator(time_obj):
            glu_val = nutr_labels.index("glucose")
            gal_val = nutr_labels.index("galactose")
            vals = []
            for t in time_obj.t:
                # use 60% glucose, 40% galactose
                if np.random.rand() <= 0.6:
                    vals.append(glu_val)
                else:
                    vals.append(gal_val)
            return vals
        nutr_simulator = glu_gal_simulator
        num_iters = 30
        # compare results of the two policies across iterations
        bh_finals = []
        pop_finals = []
        for n in xrange(num_iters):
            print "n = %d" %(n)
            # simulate environment
            env_obj = \
              env.MixedDiscEnvironment(nutr_labels, nutr_simulator,
                                       nutr_growth_rates=nutr_growth_rates,
                                       mismatch_growth_rate=mismatch_growth_rate)
            env_obj.simulate(time_obj)
            print "env: "
            print env_obj.hist
            ## bet-hedging policy
            bh_policy_obj = \
              policies.GrowthPolicy(policy_func=policies.bh_particle_filter_policy,
                                    policy_params=params)
            bh_growth_obj = growth.Growth(init_pop_sizes,
                                          env_obj=env_obj,
                                          policy_obj=bh_policy_obj)
            print "Bet-hedging results: "
            bh_results = bh_growth_obj.simulate(time_obj)
            bh_final_popsize = bh_results["pop_sizes"][-1].sum()
            bh_finals.append(bh_final_popsize)
            print "response: "
#            print bh_results
            print bh_results["prob_nutrient_states"]
            print " - final: ", bh_final_popsize
            print " --- " * 50
            ## non-bet-hedging policy
            pop_policy_obj = \
              policies.GrowthPolicy(policy_func=policies.posterior_pred_policy,
                                    policy_params=params)
            pop_growth_obj = growth.Growth(init_pop_sizes,
                                           env_obj=env_obj,
                                           policy_obj=pop_policy_obj)
            pop_results = pop_growth_obj.simulate(time_obj)
            print "Non-bet-hedging results: "
            pop_final_popsize = pop_results["pop_sizes"][-1].sum()
            pop_finals.append(pop_final_popsize)
            print "results: "
            print pop_results["prob_nutrient_states"]
            print " - final: ", pop_final_popsize
        # compare final population sizes
        bh_finals = np.array(bh_finals)
        pop_finals = np.array(pop_finals)
        print "bh finals: ", np.mean(bh_finals), np.std(bh_finals) / num_iters
        print "pop finals: ", np.mean(pop_finals), np.std(pop_finals) / num_iters
        
        
class TestFitness(unittest.TestCase):
    """
    Test fitness simulation.
    """
    def test_a_ssm_simulator(self):
        print "test ssm glucose-galactose simulator"
        p_switch_to_switch = 0.10
        p_noswitch_to_switch = 0.95
        p_init_output = 0.5
        p_init_switch = 0.5
        num_iters = 5
        time_obj = time_unit.Time(1, 150, 1)
        out_trans_mat1 = np.array([[0., 1],
                                   [1., 0]])
        out_trans_mat2 = np.array([[1., 0.],
                                   [1., 0.]])
        for n in xrange(num_iters):
            print "iter = %d" %(n)
            data = \
              nutrient_simulators.ssm_nutrient_simulator(time_obj,
                                                         out_trans_mat1=out_trans_mat1,
                                                         out_trans_mat2=out_trans_mat2,
                                                         p_switch_to_switch=p_switch_to_switch,
                                                         p_noswitch_to_switch=p_noswitch_to_switch,
                                                         p_init_switch=p_init_switch,
                                                         p_init_output=p_init_output)
        
    def test_fitness_policies(self):
        """
        Compare several policies to each other.
        """
        ###
        ### TODO: finish this. Make environment with mostly glu
        ###
        # set seed
        np.random.seed(20)
        t_start = 0
        t_end = 100
        t_step = 1
        time_obj = time_unit.Time(t_start, t_end, t_step)
        # choose an environment that gives mostly glucose states
        # and see if it leads to roughly similar population size
        # that you'd get with glucose-only growth
        out_trans_mat1 = np.array([[0., 1],
                                   [1., 0]])
        out_trans_mat2 = np.array([[1., 0.],
                                   [1., 0.]])
        p_switch_to_switch = 0.10
        p_noswitch_to_switch = 0.10
        p_init_output = 0.5
        p_init_switch = 0.5
        data = \
          nutrient_simulators.ssm_nutrient_simulator(time_obj,
                                                     out_trans_mat1=out_trans_mat1,
                                                     out_trans_mat2=out_trans_mat2,
                                                     p_switch_to_switch=p_switch_to_switch,
                                                     p_noswitch_to_switch=p_noswitch_to_switch,
                                                     p_init_switch=p_init_switch,
                                                     p_init_output=p_init_output)
        data = np.array(data)
        print "data: ", data
        num_timesteps = len(data)
        nutr_growth_rates = [0.3, 0.075]
        mismatch_growth_rate = 0.
        nutr_labels = ["glucose", "galactose"]
        num_nutrs = len(nutr_labels)
        def nutr_simulator(time_obj):
            return data
        # create environment
        env_obj = \
          env.MixedDiscEnvironment(nutr_labels, nutr_simulator,
                                   nutr_growth_rates=nutr_growth_rates,
                                   mismatch_growth_rate=mismatch_growth_rate)
        # simulate the environment
        env_obj.simulate(time_obj)
        # policies to run on environment
        all_policies = OrderedDict()
        all_policies["Posterior predictive"] = policies.posterior_pred_policy
#        all_policies["Plastic"] = policies.plastic_growth_policy
#        all_policies["Random"] = policies.rand_growth_policy
        all_policies["Glucose-only"] = policies.glu_only_growth_policy
        # run fitness simulation with these policies
        init_pop_sizes = [100, 0]
        params = {"nutr_labels": nutr_labels,
                  "nutr_growth_rates": nutr_growth_rates,
                  "mismatch_growth_rate": mismatch_growth_rate,
                  "decision_lag_time": 1,
                  "init_pop_sizes": init_pop_sizes,
                  "num_sim_iters": 1,
                  # policy-related parameters
                  "num_switch_states": 2,
                  "num_outputs": num_nutrs}
        fitness_obj = fitness.FitnessSim(all_policies, env_obj, params)
        fitness_results = fitness_obj.simulate(time_obj)
        print "final population sizes: "
        final_time = fitness_results[fitness_results["t"] == time_obj.t[-1]]
        # compare the theoretical attainable population size
        # based on deterministic growth model in glucose
        # to the result we get: they should be similar.
        post_pred_results = final_time[final_time["policy"] == "Posterior predictive"]
        post_pred_popsize = post_pred_results["pop_size"]
            

        
if __name__ == "__main__":
    unittest.main()


