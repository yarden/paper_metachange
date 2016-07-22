##
## Book keeping for simulation
##
import os
import sys
import time
import glob
import ast
import json
import cPickle as pickle

import pandas
import numpy as np

from collections import OrderedDict
import utils

LAST_RUN_FNAME = "last_run.txt"
PARAMS_FNAME = "params.txt"

def get_sim_time():
    return time.strftime("%m-%d-%Y_%I:%M:%S")

def load_params(fname):
    with open(fname, "r") as file_in:
        params = ast.literal_eval(file_in.read())
    return params

def load_data(fname):
    data = None
    with open(fname, "r") as file_in:
        data = pickle.load(file_in)
    return data

def save_data_as_df(data, fname, sep="\t"):
    """
    Save data as pandas dataframe.
    """
    df = data
    if type(data) != pandas.DataFrame:
        # convert to dataframe if it isn't already
        df = pandas.DataFrame(data)
    df.to_csv(fname, index=False, sep=sep)

class Simulation:
    """
    simulation results.

    Args:
    - name: name of simulation
    - params: parameters to be used
    - data: key value pairs, where each key represents
      a section of the simulation to be saved as filename
      and the value is a dataframe to populate that file.
      Example:

        {'sim1': df1,
         'sim2': df2}

      will get written as "sim1.txt" with df1 as
      contents, and df2 as contents for "sim2.txt".
    """
    def __init__(self, name=None, params=None, from_dir=None):
        self.name = name
        self.params = params
        self.data = OrderedDict()
        self.last_run_time = None
        self.run_fname = None
        self.params_fname = None
        self.sim_dirname = None
        self.from_dir = from_dir
        self.sep = "\t"
        if self.from_dir is not None:
            self.load(self.from_dir)

    def add_data(self, name, df):
        self.data[name] = df

    def save(self, dirname, with_index=True):
        """
        save results to directory.
        """
        utils.make_dir(dirname)
        # simulation directory
        self.sim_dirname = os.path.join(dirname, self.name)
        utils.make_dir(self.sim_dirname)
        self.run_fname = os.path.join(self.sim_dirname, LAST_RUN_FNAME)
        # write simulation time
        self.last_run_time = get_sim_time()
        with open(self.run_fname, "w") as run_f:
            run_f.write("%s" %(self.last_run_time))
        # write parameters as JSON
        self.params_fname = os.path.join(self.sim_dirname, PARAMS_FNAME)
        with open(self.params_fname, "w") as params_out:
            json.dump(self.params, params_out)
        # write data as dataframe
        for name in self.data:
            fname = os.path.join(self.sim_dirname, "%s.sim" %(name))
            self.data[name].to_csv(fname, sep=self.sep, index=with_index)
        return self.sim_dirname

    def load(self, sim_dirname):
        if not os.path.isdir(sim_dirname):
            raise Exception, "No simulation in %s" %(sim_dirname)
        self.name = os.path.basename(sim_dirname)
        self.sim_dirname = sim_dirname
        self.run_fname = os.path.join(self.sim_dirname, LAST_RUN_FNAME)
        if not os.path.isfile(self.run_fname):
            raise Exception, "No run file found for simulation in %s" \
                             %(sim_dirname)
        # load last run time
        with open(self.run_fname) as run_f:
            self.last_run_time = run_f.readline().strip()
        # load parameters
        self.params_fname = os.path.join(self.sim_dirname, PARAMS_FNAME)
        if not os.path.isfile(self.params_fname):
            raise Exception, "No params file found for simulation in %s" \
                             %(sim_dirname)
        with open(self.params_fname) as params_f:
            self.params = json.load(params_f)
        # load simulation contents
        sim_files = glob.glob("%s/*.sim" %(self.sim_dirname))
        print "loading simulation (%d files)" %(len(sim_files))
        for curr_fname in sim_files:
            curr_fname = utils.pathify(curr_fname)
            sim_name = os.path.basename(curr_fname).split(".sim")[0]
            sim_df = pandas.read_table(curr_fname, sep=self.sep)
            self.data[sim_name] = sim_df

            
if __name__ == "__main__":
    sim_obj = Simulation(name="my_sim", params={"a": [1,2,3], "b": 0.5})
    # write simulation
    sim_obj.add_data("sim1", pandas.DataFrame({"foo": ["a", "b", "c"],
                                               "bar": [10, 20, 30]}))
    sim_obj.add_data("sim2", pandas.DataFrame({"quux": ["hello", "there"]}))
    sim_obj.save("./simtest")
    # read simulation
    new_sim = Simulation(from_dir="./simtest/my_sim")
    print new_sim.params
    print new_sim.data








