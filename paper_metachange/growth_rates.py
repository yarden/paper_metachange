##
## analyzing growth rates
##
import os
import sys
import time

import scipy
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pylab as plt
import pandas
import paths
import utils

def get_spline_slopes(x, y, kind="linear", num_bins=20,
                      slope_x_start=None,
                      slope_x_end=None):
    """
    Fit spline to get slopes.
    """
    if slope_x_start is None:
        slope_x_start = x.min()
    if slope_x_end is None:
        slope_x_end = x.max()
    fit = interp1d(x, y, kind=kind)
    x_coarse = np.linspace(x.min(), x.max(), num_bins)
    # find the maximum slope
    # use only specified range for slopes
    inds = (x_coarse >= slope_x_start) & (x_coarse <= slope_x_end)
    x_pairs = utils.rolling_window(x_coarse[inds], 2)
    y_coarse = fit(x_coarse)
    slopes = map(lambda pair: pair[-1] - pair[0],
                 utils.rolling_window(y_coarse[inds], 2))
    slopes = np.array(slopes)
    print slopes, " <<< "
    results = {"x_coarse": x_coarse,
               "y_coarse": y_coarse,
               "y_slopes": slopes,
               "x_pairs": x_pairs,
               "fit": fit}
    return results
    
def load_data():
    exp_label = "Jun6_2016"
    plate_fname = paths.ENVISION_DATA[exp_label]["data"]
    labels_fname = paths.ENVISION_DATA[exp_label]["labels"]
    df, rawdata = envision.parse_envision_plate(plate_fname, labels_fname,
                                                normalize=True)
    sampling_time_in_min = 5.
    df["time"] = df["time"] * (sampling_time_in_min/60.)
    # beginning of growth curve (exclude all time points
    # before this; strip out the lag phase)
    growth_start_in_hr = 0.
    growth_end_in_hr = 20.
    print "keeping only data between %.2f and %.2f" %(growth_start_in_hr,
                                                      growth_end_in_hr)
    df = df[df["time"] >= growth_start_in_hr]
    df = df[df["time"] <= growth_end_in_hr]
    samples_to_plot = ["Gal"]
    df = df[df["sample"].isin(samples_to_plot)]
    od_df = df[df["channel"] == "OD600(1)"]
    # blank subtraction
    od_df["value"] -= 0.03
    od_df["value"] = map(lambda x: max(x, 0.03), od_df["value"])
    # pick only select wells
    wells_to_plot = ["C1"]
    od_df = od_df[od_df["well_id"].isin(wells_to_plot)]
    return od_df

def main():
    pass

if __name__ == "__main__":
    main()
