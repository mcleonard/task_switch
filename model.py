""" Module which contains all the model code. """

from collections import namedtuple
import numpy as np

import constants
from hopfield import Hopfield

Stimulus = namedtuple('Stimulus', ['word', 'color'])

class state(object):
    cue = 'word'
    coeffs = (0.95, 0.3)

def init_network(coeffs=(0.95, 0.3)):
    network = Hopfield()
    network.W = calculate_weights(coeffs)
    return network

def calculate_weights(coeffs):
    """ Calculate the weight matrix for a given tuple of coefficients. """
    
    W = coeffs[0]*constants.W_wrd + coeffs[1]*constants.W_clr
    
    # de-meanify the W matrix.  This reduces the energy for the all on and all
    # off spurious states, so we don't end up in them all the time.
    W = W - 0.5*np.tile(W.mean(axis=0), (16,1))

    return W

def random_stimuli(size=1, congruent=-1, neutral=0):
    """ Generate random word/color pairs. 

        Parameters
        ----------
        congruent:
            1 for all congruent stimuli
            0 for both congruent and incongruent stimuli
            -1 for no congruent stimuli
        neutral:
            'color' for neutral colors
            'word' for neutral words
            1 for both neutral and non-neutral stimuli
            0 for no neutral stimuli
    """

    if neutral == 1:
        colors = ['red', 'blue', 'green', 'none']
    elif neutral in [0, 'color', 'word']:
        colors = ['red', 'blue', 'green']

    if congruent == 1:
        rand_colors = np.random.choice(colors, (1, size))
        pairs = np.concatenate([rand_colors]*2).T
    elif congruent == 0:
        pairs = np.random.choice(colors, (2, size)).T
    elif congruent == -1:
        rand_colors = [np.random.choice(colors, (2,1), replace=False).T 
                       for i in range(size)]
        pairs = np.concatenate(rand_colors)

    if neutral == 'word':
        pairs[:,0] = 'none'
    elif neutral == 'color':
        pairs[:,1] = 'none'

    stimuli = [ Stimulus(*pair) for pair in pairs ]
    return stimuli

def gen_stimulus(word, color):
    """ Generate the stimulus state for word and color """
    # Find the correct stimulus unit.
    index = np.where((constants.words[word] + 
                      constants.colors[color]) > 0 )
    # Create a state with all units off.
    state = -np.ones(len(constants.words[word]))
    # Turn on the unit corresponding to the stimulus.
    state[index] = 1
    return state

def random_blocks(num_blocks, word_len, color_len):
    """ Generate random length blocks, sampled from Poisson distributions,
        for use in a simulation. 
    """
    from scipy.stats import poisson
    # Random variable sampling from a Poisson distribution
    block_length = poisson.rvs
    words = block_length(word_len, size=num_blocks/2)
    colors = block_length(color_len, size=num_blocks/2)
    blocks = np.array([[words], [colors]])
    blocks = blocks.reshape((num_blocks,), order='F')

    return blocks

def update_W(word_rate=0.06, color_rate=0.03,
             trial_interval=1, prep_time=0, 
             word_limits=(0.05, 0.95),
             color_limits=(0.3, 0.7),
             relative=False):
    """ Update the network's weight matrix W. """
    
    coeff_rates = np.array([word_rate, color_rate])
    # Check the cue to see what we need to do
    if state.cue == 'word':
        coeffs_change = coeff_rates * np.array([1, -1])
    elif state.cue == 'color':
        coeffs_change =  coeff_rates * np.array([-1, 1])

    coeffs = np.array(state.coeffs)
    
    # Additive, maybe multiplcative would be better?  
    # Or some other function?
    new_coeffs = coeffs + (prep_time + trial_interval) * coeffs_change
    
    # Make sure coefficients are within limits
    if new_coeffs[0] > word_limits[1]:
        new_coeffs[0] = word_limits[1]
    elif new_coeffs[0] < word_limits[0]:
        new_coeffs[0] = word_limits[0]

    if new_coeffs[1] > color_limits[1]:
        new_coeffs[1] = color_limits[1]
    elif new_coeffs[1] < color_limits[0]:
        new_coeffs[1] = color_limits[0]

    if relative:
        new_coeffs = new_coeffs/np.sum(new_coeffs)

    state.coeffs = new_coeffs
    network.W = calculate_weights(new_coeffs)

    return new_coeffs

def give_cue(cue):
    """ Cues the model to one task. cue can be 'word', 'color', or 'switch'.
    """
    if cue == 'switch':
        if state.cue == 'word':
            new_cue = 'color'
        elif state.cue == 'color':
            new_cue = 'word'

        state.cue = new_cue
    else:
        state.cue = cue

def outcome_proportion(outcome, outcome_record):
    """ Calculate the proportion of a particular outcome in a simulated 
        session. Valid outcomes are 'hits', 'errors', 'hesitations'.
    """

    outcome_dict = dict(zip(['hits', 'errors', 'hesitations'], [1, 0, -1]))
    value = outcome_dict[outcome]
    N_outcomes = np.sum(np.array(outcome_record)==value)
    N_trials = len(outcome_record)
    
    try:
        proportion = N_outcomes / N_trials
    except ZeroDivisionError as e:
        raise e('No outcomes yet.  Run a simulation.')

    return proportion

def response_outcome(word, color, response):
    """ Check if the response is correct, wrong, or just a spurious state. """

    # First, let's identify the response.
    for each in [constants.colors, constants.words]:
        for key, val in each.items():
            if (response == val).all():
                result = key

    if 'result' in locals():
        pass
    else:
        result = 'spurious'

    # Now, let's get the expected result
    if state.cue == 'word':
        expected_result = word
    elif state.cue == 'color':
        expected_result = color

    if result == 'spurious':
        outcome = -1
    elif result != expected_result:
        outcome = 0
    elif result == expected_result:
        outcome = 1

    return outcome

def reaction_time(interval, hesitation_rate):
    """ Calculate the average reaction time for a given trial interval 
        and hesitation rate. 
    """
    return interval/(1-hesitation_rate)

#============================================================================#

network = init_network()


