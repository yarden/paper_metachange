##
## Utilities for working with probabilities
##
import numpy as np

def sample_trans_prob(prev_value,
                      p_true_to_true,
                      p_false_to_true):
    """
    Given previous value and transition probabilities of True -> True
    and False -> True, sample value for transition.
    """
    assert(type(prev_value) == bool), "Previous value has to be boolean."
    rand_value = np.random.rand()
    if prev_value:
        return (rand_value <= p_true_to_true)
    else:
        return (rand_value <= p_false_to_true)

def sample_binary_state(prob_state):
    """
    Interpret probability as probability of the 0 state.
    """
    probs = [prob_state, 1. - prob_state]
    return np.random.multinomial(1, probs).argmax()
 
def sample_multinomial(num_draws, probs):
    """
    Sample multinomial outcomes given a vector of probabilities.
    Return a vector of 'num_draws'-many draws, where i-th
    entry is the state (0-based index in probs) that was drawn.
    """
    #### vectorized version
    result = (np.random.multinomial(1, probs, size=num_draws) == 1).argmax(1)
    return result
    # outcomes = np.zeros(num_draws, dtype=np.int32)
    # for n in xrange(num_draws):
    #     draws = np.random.multinomial(1, probs)
    #     outcomes[n] = np.where(draws == 1)[0][0]
    # return outcomes


def sample_markov_chain(num_samples, init_probs, trans_mat):
    """
    Sample from a Markov chain.

    Args:
    -----
    - init_probs: initial state probabilities. Expects
    np.array
    - trans_mat: transition matrix, where rows sum to 1.
    Expects np.matrix
    """
    samples = np.zeros(num_samples, dtype=np.int32)
    # sample initial state as vector representation
    prev_state = np.random.multinomial(1, init_probs)
    # get state index (0-based)
    prev_state_ind = (prev_state == 1).argmax()
    samples[0] = prev_state_ind
    for n in xrange(1, num_samples):
        next_state_probs = prev_state.dot(trans_mat)
        next_state = np.random.multinomial(1, next_state_probs)
        next_state_ind = (next_state == 1).argmax()
        samples[n] = next_state_ind
        prev_state = next_state
    return samples
    

def flip(p):
    return (np.random.rand() <= p)

def flip_int(p):
    return int(np.random.rand() <= p)
        
