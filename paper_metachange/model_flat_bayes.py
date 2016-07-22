##
## Simple "flat" Dirichlet-Multinomial model where we analytically
## get predictive distribution.
##
## This is used as a kind of null model for meta-changing environments.
##
import os
import sys
import time

import numpy as np

import particlefever
import particlefever.distributions as distributions

import prob_utils

class FlatBayesModel:
    """
    Dirichlet prior on outcomes with multinomial
    likelihood. Posterior predictive distribution
    has analytic form.
    """
    def __init__(self, num_states, alpha_prior=None):
        self.num_states = num_states
        assert (self.num_states == 2), "Only two states supported."
        self.alpha_prior = alpha_prior
        if self.alpha_prior is None:
            self.alpha_prior = np.array([1., 1.])
        # matrix of transition counts
        # previous state, columns represent next states, i.e.
        # trans_mat[i, j] is number of times we transitioned
        # from state i to state j
        self.trans_mat = np.zeros((self.num_states, self.num_states))

    def __str__(self):
        return "FlatBayesModel(num_states=%d, alpha_prior=%s, trans_mat=%s)" \
               %(self.num_states,
                 str(self.alpha_prior),
                 str(self.trans_mat))

    def __repr__(self):
        return self.__repr__()

    def predict(self, data, lag=1):
        """
        Return posterior predictive probability of first state
        (since we're assuming only binary states) for each data i
        point, assuming that we've seen the i-1 data points.
        """
        assert (lag == 1), "Only lag = 1 is supported."
        # get probability of initial state from the prior
        prev_counts = np.zeros(self.num_states)
        dist_obj = distributions.DirMultinomial(prev_counts, self.alpha_prior)
        prior_probs = np.exp(dist_obj.log_posterior_pred)
        preds = [prior_probs]
        if len(data) == 1:
            return preds
        prev_state = data[0]
        # for next states, use posterior predictive distribution,
        # and update it for each data point
        ind = 1
        for next_state in data[1:]:
            # predict state based on posterior predictive
            # distribution. the previous state tells us what
            # posterior predictive distribution to look at
            prev_counts = self.trans_mat[prev_state, :]
            print self.trans_mat
            dist_obj = distributions.DirMultinomial(prev_counts,
                                                    self.alpha_prior)
            next_pred = np.exp(dist_obj.log_posterior_pred)
            preds.append(next_pred)
            # update the transition counts
            self.trans_mat[prev_state, next_state] += 1
            # update previous state
            prev_state = next_state
        preds = np.array(preds)
        assert (len(preds) == len(data)), \
               "Need as many predictions as data points."
        return preds

def main():
    data = np.array([0, 1] * 10 + [0, 0] * 10)
    print "data: "
    print data
    num_states = 2
    model = FlatBayesModel(num_states)
    print "predictions: "
    preds = model.predict(data)
    print preds


if __name__ == "__main__":
    main()
