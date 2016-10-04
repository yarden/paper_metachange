##
##
##
import numpy as np
import pandas
import utils
import fitness
import simulation
import scipy
from scipy.interpolate import interp1d
import matplotlib.pylab as plt

f = "../simulations_data/sudden_switch/merged_switch_ssm_fitness_sims.data"

sim_info = simulation.load_data(f)
results = sim_info["data"]
# get the simulations that should be plotted based on their
# parameter values
params_to_plot = [{"p_switch_to_switch": 0.1,
                   "p_noswitch_to_switch": 0.1},
                  {"p_switch_to_switch": 0.1,
                   "p_noswitch_to_switch": 0.95},
                  {"p_switch_to_switch": 0.95,
                   "p_noswitch_to_switch": 0.1},
                  {"p_switch_to_switch": 0.95,
                   "p_noswitch_to_switch": 0.95}]
sims_to_plot = []
for sim_name in results:
    # see if current simulation matches the parameters
    # we're looking for
    for curr_params in params_to_plot:
        if len(results[sim_name]["params"]["nutr_labels"]) != 2:
            # skip any simulation that doesn't have two nutrients
            continue
        if utils.all_match(curr_params, results[sim_name]["params"]):
            sims_to_plot.append(sim_name)

def get_spline_fit(x, y):
    """
    Get spline fit.
    """
    spl = scipy.interpolate.UnivariateSpline(x, y, k=4, s=1)
    return spl(x)

def get_spline_derivs_fit(x, y):
    """
    Get spline derivatives fit.
    """
    spl = scipy.interpolate.UnivariateSpline(x, y, k=4, s=1)
    derivs = spl.derivative()
    return derivs(x)

sim_to_plot = sims_to_plot[0]            
df = results[sim_to_plot]["data"]
popsizes = fitness.str_popsizes_to_array(df["log_pop_sizes"])
# take the total population size: sum of populations tuned to any
# nutrient state
df["log2_pop_size"] = np.log2(np.exp(popsizes).sum(axis=1))
print df
g = df.groupby(["policy", "sim_num"])
new_df = []
for name, group in g:
    curr_df = group.copy()
    print "fitting to: ", curr_df["t"], curr_df["growth_rates"]
    curr_df["fitted_growth_rate"] = \
      get_spline_fit(curr_df["t"], curr_df["growth_rates"])
    curr_df["fitted_growth_rate_from_pop"] = \
      get_spline_derivs_fit(curr_df["t"], curr_df["log2_pop_size"])
    plt.figure()
    plt.plot(curr_df["t"], curr_df["growth_rates"])
    plt.plot(curr_df["t"], curr_df["fitted_growth_rate"], color="r")
    plt.plot(curr_df["t"], curr_df["fitted_growth_rate_from_pop"], color="b")
    plt.savefig("test.pdf")
    print curr_df
    raise Exception
    new_df.append(curr_df)

new_df = pandas.concat(new_df)
print new_df.head()


raise Exception, "test"
print " --- \n"
print "x: ", x
print x["log2_pop_size"]
print x["t"]
spl = scipy.interpolate.UnivariateSpline(x["t"], x["log2_pop_size"], k=4, s=1)
print "spl: ", spl
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x["t"], x["log2_pop_size"])
#f2 = interp1d(x["t"], x["log2_pop_size"], kind='cubic')
#plt.plot(x["t"], spl(x["t"]), ":", linewidth=0.5, color="r")
#plt.plot(x["t"], f2(x["t"]), ":", linewidth=0.5, color="r")
plt.plot(x["t"], spl(x["t"]), ":", linewidth=0.5, color="r")
plt.subplot(2, 1, 2)
#plt.plot(x["t"], x["log2_pop_size"])
y = spl.derivative()
t = x["t"]
gr = x["growth_rates"]
#fit = interp1d(x["t"], gr, kind="cubic")
#plt.plot(x["t"], gr, "k")
# get growth rate from the fitted population size
fitted_y = scipy.interpolate.UnivariateSpline(x["t"], x["log2_pop_size"], k=4, s=1)
# now take derivative of spline
d = fitted_y.derivative()
# growth rates from pop size
plt.plot(x["t"], d(x["t"]), "-o", label="From popsize")
# growth rates from fitted raw growth rates
fitted_gr = scipy.interpolate.UnivariateSpline(x["t"], x["growth_rates"], k=5, s=0.5)
plt.plot(x["t"], fitted_gr(x["t"]), "-o", color="b", label="From gr")
# raw growth rates
plt.plot(x["t"], x["growth_rates"], "r", label="Raw")
plt.legend()
#plt.ylim([0, 0.5])
plt.savefig("test.pdf")


#x = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16])
#y = np.array([0, 2, 1, 10, 5, 6, 7, 8, 9, 10])
#spl = scipy.interpolate.UnivariateSpline(x, y, k=4, s=1)
#plt.figure()
#plt.savefig("test2.pdf")

