##
## Paths
##
import os
import sys
import time
import utils

##
## Main directory of code -- must be set 
##
MAIN_DIR = \
  os.path.join(os.path.dirname(os.path.join(os.path.realpath(__file__))),
               "..")
MAIN_DIR = os.path.abspath(MAIN_DIR)

print "main code directory: %s" %(MAIN_DIR)

PIPELINES_INFO = os.path.join(MAIN_DIR, "pipelines_info")
utils.make_dir(PIPELINES_INFO)

PIPELINE_START_FNAME = os.path.join(PIPELINES_INFO, "paper_pipeline.start")

# dummy file to start sudden pipeline  
PIPELINE_SUDDEN_START_FNAME = os.path.join(PIPELINES_INFO,
                                           "pipeline_sudden.start")

##
## Data files
##
# Data directory
DATA_DIR = os.path.join(MAIN_DIR, "data")

# Data for Liti-Fay growth rates
LITI_FAY_DIR = os.path.join(DATA_DIR, "liti_fay_growthrates")

# Plots directory
PLOTS_DIR = os.path.join(MAIN_DIR, "plots")
utils.make_dir(PLOTS_DIR)

# Simulations plots
SIM_PLOTS_DIR = os.path.join(PLOTS_DIR, "simulations")
utils.make_dir(SIM_PLOTS_DIR)

RUFFUS_PLOTS_DIR = os.path.join(PLOTS_DIR, "ruffus")
utils.make_dir(RUFFUS_PLOTS_DIR)

SIM_DATA_DIR = os.path.join(MAIN_DIR, "simulations_data")
utils.make_dir(SIM_DATA_DIR)

SIM_PARAMS_DIR = os.path.join(MAIN_DIR, "simulations_params")

# sudden switch environments
SUDDEN_DATA_DIR = os.path.join(SIM_DATA_DIR, "sudden_switch")
utils.make_dir(SUDDEN_DATA_DIR)
