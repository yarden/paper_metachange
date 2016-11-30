##
## Utilities for working with and solving SBML
##
import os
import sys
import time

from collections import OrderedDict

import numpy as np
import pandas

import matplotlib.pylab as plt
import seaborn as sns

try:
    import roadrunner
except ImportError:
    raise Exception, "Need roadrunner library to run."

def set_model_vars(rr_model, vars_to_vals):
    """
    Set model values to the value at the *index*
    t_ind.
    
    Args:
    - rr_model: roadrunner model
    - t_ind: index of time to use, e.g. t = 1
    - vars_to_vals: mapping from variables to their values
    """
    for var in vars_to_vals:
        if var not in rr_model.keys():
            raise Exception, "Variable %s not in model." %(var)
        rr_model[var] = vars_to_vals[var]
    return rr_model

def get_rr_results_as_dict(rr_results, t_ind=-1):
    """
    Get the results of an roadrunner model simulation
    as dictionary (this is used to convert it to pandas
    DataFrame later on.) By default, takes the last
    value of the simulation.

    Args:
    - rr_results: roadrunner results

    Kwargs:
    - t_ind: time index to use. By default uses the last one
      (i.e. t_ind=-1).
    """
    data = {}
    for c in rr_results.colnames:
        data[c] = rr_results[c][t_ind]
    return data

class DoseSched:
    """
    Dose scheduler for SBML model.
    """
    def __init__(self, t_start, t_end, num_steps):
        self.t_start = t_start
        self.t_end = t_end
        self.num_steps = num_steps
        self.times = np.linspace(self.t_start, self.t_end, self.num_steps)
        self.doses = []
        # initial dose data
        self.dose_data = OrderedDict()
        for t_ind in xrange(len(self.times)):
            self.dose_data[t_ind] = {}

    def __str__(self):
        return "DoseSched(t_start=%.1f, t_end=%.1f, num_steps=%d,\n" \
               "doses=%s)" %(self.t_start, self.t_end, self.num_steps,
                            str(self.doses))

    def __repr__(self):
        return self.__str__()

    def find_close_t(self, t):
        """
        Find closest time to the given time based on
        our binning. Return the time and its bin.
        """
        close_t_ind = np.abs(self.times - t).argmin()
        close_t = self.times[close_t_ind]
        return close_t, close_t_ind

    def add_dose(self, var_name, dose_start, dose_end, var_val):
        t_start, t_start_ind = self.find_close_t(dose_start)
        t_end, t_end_ind = self.find_close_t(dose_end)
        self.doses.append((var_name, var_val, dose_start, dose_end))
        for curr_t_ind in xrange(t_start_ind, t_end_ind + 1):
            curr_t = self.times[curr_t_ind]
            self.dose_data[curr_t_ind][var_name] = var_val
        
    def get_doses_data(self):
        """
        Return doses as an OrderedDict.
        """
        return self.dose_data

class SBML:
    """
    Solving SBML model.
    """
    def __init__(self, model_fname):
        if not os.path.isfile(model_fname):
            raise Exception, "File %s not found." %(model_fname)
        self.model_fname = model_fname
        self.model = None

    def simulate_with_doses(self, times, dose_sched):
        """
        Simulate model with doses given.

        Args:
        - times: array of time to simulate
        - time_to_var_vals: mapping from time to variable values
        - variables to track: list of variable names to track

        returns a list of roadrunner result objects.
        """
        # get the results for initial time 
        model = roadrunner.RoadRunner(self.model_fname)
        ### TODO: REFACTOR THIS PART TO get_initial_values()
        dose_data = dose_sched.dose_data
        model = set_model_vars(model, dose_data[0])
        print model.keys, " << < "
        duration = times[1] - times[0]
        print "duration: ", duration
        assert (duration > 0), "Time step size must be positive."
        result = model.simulate(start=times[0], end=times[-1],
                                duration=duration,
                                steps=1)
        # get the results from initial time
        all_results = [get_rr_results_as_dict(result, t_ind=0)]
        prev_t = times[0]
        t_ind = 1
        for next_t_ind in xrange(1, len(times)):
            model = set_model_vars(model, dose_data[next_t_ind])
            next_t = times[next_t_ind]
            # simulate one duration-sized step ahead
            result = model.simulate(start=prev_t, end=next_t,
                                    duration=duration, steps=1)
            # take the result
            results_dict = get_rr_results_as_dict(result, t_ind=-1)
            # ensure that results match the values we're setting
            for var_name in dose_data[next_t_ind]:
                # use concentration notation 
                results_dict["[%s]" %(var_name)] = dose_data[next_t_ind][var_name]
                #results_dict["[%s]" %(var_name)] = dose_data[next_t_ind][var_name]
            all_results.append(results_dict)
            # advance time value
            prev_t = next_t
            # advance time index
            t_ind += 1
        return pandas.DataFrame(all_results)
            
def main():
    f = "./sbml_models/glu_gal_transition_counter.xml"
    sbml_model = SBML(f)
    t_start = 0
    t_end = 600
    times = np.linspace(t_start, t_end, 1000)
    doser = DoseSched(t_start, t_end, 1000)
    # Glu
    doser.add_dose("Glu", 0, 99, 100)
    # Gal
    doser.add_dose("Gal", 99, 199, 100)
    # Glu
    doser.add_dose("Glu", 199, 299, 100)
    # Gal
    doser.add_dose("Gal", 299, 399, 100)
    #print doser
    results = sbml_model.simulate_with_doses(times, doser)
    plt.figure()
    sns.set_style("ticks")
    #vars_to_plot = ["[Glu]", "[Gal]"]
    #vars_to_plot = ["[Glu_Sensor]", "[Gal_Sensor]", "[Gal_Activator]"]
    #vars_to_plot = ["[Glu_to_Gal]", "[Gal_to_Glu]"]
    vars_to_plot = ["[Glu]", "[Gal]"]
    offset = 0
    x_offset = 5
    for c in vars_to_plot:
        if c != "time":
            plt.plot(results["time"] + offset, results[c], label=c,
                     linewidth=3,
                     clip_on=False)
        offset += x_offset
    sns.despine(trim=True, offset=3.)
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    ylims = plt.gca().get_ylim()
    plt.ylim([-0.01, ylims[1]])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


# Need a better representation for values of variables
# across time.

# mapping from times to dictionary of values
# e.g.
#
# OrderedDict()


#raise Exception

# f = "/Users/yarden/my_projects/suddenswitch_paper/paper/matlab_models/simple_model.xml"
# sbml_model = SBML(f)
# times = np.linspace(0, 10, 1000)
# doser = DoseSched(0, 10, 1000)
# doser.add_dose("A", 0, 5, 50)
# doser.add_dose("B", 0, 1, 50)
# doser.add_dose("B", 1, 2, 40)
# results = sbml_model.simulate_with_doses(times, doser)
# print results
# plt.figure()
# sns.set_style("ticks")
# for c in results.columns:
#     if c != "time":
#         plt.plot(results["time"], results[c], label=c,
#                  clip_on=False)
# sns.despine(trim=True, offset=3.)
# plt.xlabel("Time")
# plt.ylabel("Concentration")
# plt.legend()
# plt.show()

#sbml_model.model.plot()



# test case
#f = "/Users/yarden/my_projects/suddenswitch_paper/paper/matlab_models/simple_model.xml"
#rr = roadrunner.RoadRunner(f)
#t_start = 0
#t_end = 10
#rr.A = 20
#result = rr.simulate(start=t_start, end=t_end, steps=1000)
#print result
#rr.plot()
