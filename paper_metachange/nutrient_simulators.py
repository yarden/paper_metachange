##
## nutrient simulators
##
import os
import sys
import time

import numpy as np

import prob_utils

###
### nutrient_simulators
###
def simulate_switch_states(num_points,
                           p_switch_to_switch,
                           p_noswitch_to_switch,
                           p_init_switch=0.5,
                           init_state=None):
    """
    Simulate switch states.
    """
    if init_state is None:
        init_state = prob_utils.sample_binary_state(p_init_switch)
    data = [init_state]
    if num_points == 1:
        return data
    points = range(num_points) 
    for n in points[1:]:
        if data[n-1] == 0:
            # probability of transitioning from switch to switch state
            next_state = prob_utils.sample_binary_state(p_switch_to_switch)
        else:
            next_state = prob_utils.sample_binary_state(p_noswitch_to_switch)
        data.append(next_state)
    return data

# def ssm_gluc_galac_simulator(time_obj,
#                              out_trans_mat1=[[0., 1],
#                                              [1., 0]],
#                              out_trans_mat2=[[1., 0.],
#                                              [1., 0.]],
#                              p_switch_to_switch=0.8,
#                              p_noswitch_to_switch=0.2,
#                              p_init_switch=0.5,
#                              p_init_output=0.5):
#     """
#     Glucose-galactose simulator that depends on switching
#     between periodic and non-periodic states.
#     """
#     # two transition matrices to switch between
#     ####
#     #### TODO: move these transition matrices to be
#     ####       parameters
#     ####
#     out_trans_mat1 = np.array(out_trans_mat1)
#     # constant transition matrix
#     out_trans_mat2 = np.array(out_trans_mat2)
#     trans_mats = [out_trans_mat1, out_trans_mat2]
#     num_points = len(time_obj.t)
#     # draw switch states
#     switch_states = \
#       simulate_switch_states(num_points - 1,
#                              p_switch_to_switch=p_switch_to_switch,
#                              p_noswitch_to_switch=p_noswitch_to_switch,
#                              p_init_switch=p_init_switch)
#     data = [prob_utils.sample_binary_state(p_init_output)]
#     prev_output = data[0]
#     for n in xrange(1, num_points, 1):
#         # choose which transition matrix to draw from
#         trans_mat = trans_mats[switch_states[n - 1]]
#         # draw data point
#         data_point = \
#           np.random.multinomial(1, trans_mat[prev_output, :]).argmax()
#         data.append(data_point)
#         prev_output = data_point
#     data = np.array(data)
#     return data


def ssm_nutrient_simulator(time_obj,
                           out_trans_mat1=None,
                           out_trans_mat2=None,
                           p_switch_to_switch=0.8,
                           p_noswitch_to_switch=0.2,
                           p_init_switch=0.5,
                           p_init_output=0.5):
    """
    Nutrient simulator where the generative process is
    the 2-hidden state switch SSM.

    This uses the two given output transition matrices
    (out_trans_mat1, out_trans_mat2) to do the sampling.

    Note that this only supports 2 hidden states.
    """
    # two transition matrices to switch between
    ####
    #### TODO: move these transition matrices to be
    ####       parameters
    ####
    if (out_trans_mat1 is None) or (out_trans_mat2 is None):
        raise Exception, "Expected an output transition matrix."
    out_trans_mat1 = np.array(out_trans_mat1)
    # constant transition matrix
    out_trans_mat2 = np.array(out_trans_mat2)
    trans_mats = [out_trans_mat1, out_trans_mat2]
    num_points = len(time_obj.t)
    # draw switch states
    switch_states = \
      simulate_switch_states(num_points - 1,
                             p_switch_to_switch=p_switch_to_switch,
                             p_noswitch_to_switch=p_noswitch_to_switch,
                             p_init_switch=p_init_switch)
    data = [prob_utils.sample_binary_state(p_init_output)]
    prev_output = data[0]
    for n in xrange(1, num_points, 1):
        # choose which transition matrix to draw from
        trans_mat = trans_mats[switch_states[n - 1]]
        # draw data point
        data_point = \
          np.random.multinomial(1, trans_mat[prev_output, :]).argmax()
        data.append(data_point)
        prev_output = data_point
    data = np.array(data)
    return data

