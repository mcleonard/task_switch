""" Module for running the task switching simulation. """

from collections import namedtuple
import numpy as np

import constants
import hopfield

Stimulus = namedtuple('Stimulus', ['word', 'color'])

W_coeffs_init = np.array([0.5, 0.5]) # Word, Color

class Sim(object):
    """ This object is going to hold the important task variables and 
        simulated results.
    """
    cue = []
    stimulus = []
    response = []
    outcome = []
    network = hopfield.Hopfield()
    W_coeffs =  [] # Word, Color

def reset_sim():
    """ Reset the simulation. """
    Sim.cue = []
    Sim.stimulus = []
    Sim.response = []
    Sim.outcome = []
    Sim.network = hopfield.Hopfield()
    Sim.W_coeffs = [] # Word, Color

def random_stimuli(size=1, congruent=-1, neutral=-1):
    """ Generate random word/color pairs. 

        Parameters
        ----------
        congruent:
            1 for all congruent stimuli
            0 for both congruent and incongruent stimuli
            -1 for no congruent stimuli
        neutral:
            1 for all neutral stimuli
            0 for both neutral and non-neutral stimuli
            -1 for no neutral stimuli
    """

    if neutral == 1:
        colors = ['none']
    elif neutral == 0:
        colors = ['red', 'blue', 'green', 'none']
    elif neutral == -1:
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

    stimuli = [ Stimulus(*pair) for pair in pairs ]
    return stimuli

def gen_stimulus(word, color):
    """ Generate the stimulus state for word and color """
    index = np.where((constants.words[word] + 
                      constants.colors[color]) > 0 )
    state = -np.ones(len(constants.words[word]))
    state[index] = 1
    return state

def random_blocks(num_blocks, word_len, color_len):
    """ Generate random length blocks, sampled from Poisson distributions,
        for use in a simulation. 
    """
    from scipy.stats import poisson
    block_length = poisson.rvs
    words = block_length(word_len, size=num_blocks/2)
    colors = block_length(color_len, size=num_blocks/2)
    blocks = np.array([[words], [colors]])
    blocks = blocks.reshape((num_blocks,), order='F')

    return blocks

def update_W(word_rate=0.06, color_rate=0.03,
             trial_interval=1., prep_time=0, 
             word_limits=(0.05, 0.95),
             color_limits=(0.3, 0.7),
             relative=False):
    """ Update the network's weight matrix W. """
    
    coeff_rates = np.array([word_rate, color_rate])
    # Check the cue to see what we need to do
    if Sim.cue[-1] == 'word':
        coeffs_change = coeff_rates * np.array([1, -1])
    elif Sim.cue[-1] == 'color':
        coeffs_change =  coeff_rates * np.array([-1, 1])

    if Sim.W_coeffs:
        W_coeffs = Sim.W_coeffs[-1]
    else:
        W_coeffs = W_coeffs_init
    
    # Additive, maybe multiplcative would be better?  
    # Or some other function?
    new_coeffs = W_coeffs + (prep_time + trial_interval) * coeffs_change

    # Make sure coefficients are within limits
    if new_coeffs[0] > word_limits[1]:
        new_coeffs[0] = word_limits[1]
    elif new_coeffs[0] < word_limits[0]:
        new_coeffs[0] = word_limits[0]

    if new_coeffs[1] > color_limits[1]:
        new_coeffs[1] = color_limits[1]
    elif new_coeffs[1] < color_limits[0]:
        new_coeffs[1] = color_limits[0]

    Sim.W_coeffs.append(new_coeffs)

    if relative:
        rel_coeffs = new_coeffs/np.sum(new_coeffs)
        W = (rel_coeffs[0] * constants.W_wrd +
             rel_coeffs[1] * constants.W_clr )
    else:
        W = (new_coeffs[0] * constants.W_wrd +
             new_coeffs[1] * constants.W_clr )

    # de-meanify the W matrix.  This makes the thing work.
    W = W - 0.5*np.tile(W.mean(axis=0), (16,1))

    Sim.network.W = W

def give_cue(cue):
    """ Cues the model to one task. cue can be 'word', 'color', or 'switch'.
    """
    if cue == 'switch':
        prev_cue = Sim.cue[-1]
        if prev_cue == 'word':
            new_cue = 'color'
        elif prev_cue == 'color':
            new_cue = 'word'

        Sim.cue.append(new_cue)
    else:
        Sim.cue.append(cue)

def outcome_proportion(outcome):
    """ Calculate the proportion of a particular outcome in a simulated 
        session. Valid outcomes are 'hits', 'errors', 'hesitations'.
    """

    outcome_dict =  dict(zip(['hits', 'errors', 'hesitations'], [1, 0, -1]))
    value = outcome_dict[outcome]
    N_outcomes = np.sum(np.array(Sim.outcome)==value)
    N_trials = len(Sim.outcome)
    
    try:
        proportion = N_outcomes / N_trials
    except ZeroDivisionError as e:
        raise e('No outcomes yet.  Run a simulation.')

    return proportion

def response_outcome(cue, word, color, response):
    """ Check if the response is correct, wrong, or just a spurious state. """

    # First, let's identify the response.
    for states in [constants.colors, constants.words]:
        for key, state in states.items():
            if (response == state).all():
                result = key

    if 'result' in locals():
        pass
    else:
        result = 'spurious'

    # Now, let's get the expected result
    if cue == 'word':
        expected_result = word
    elif cue == 'color':
        expected_result = color

    if result == 'spurious':
        outcome = -1
    elif result != expected_result:
        outcome = 0
    elif result == expected_result:
        outcome = 1

    Sim.outcome.append(outcome)

    return outcome

def run_trial(word, color, net_iters=30, net_temp=0.5):
    """ Simulate one trial given the word and color. """
    
    network = Sim.network
    stimulus = gen_stimulus(word, color)

    # Keeping track of the stimulus
    Sim.stimulus.append((word, color))
    
    # Now get the response from the stimulus
    network.state = stimulus
    network.run(n_iters=net_iters, init_temp=net_temp)
    response = network.state
    Sim.response.append(response)
    outcome = response_outcome(Sim.cue[-1], word, color, response)

    return response, outcome

def run_simulation(blocks, **update_kwargs):
    """ Run a simulation for the block lengths in blocks. """
    for i, block in enumerate(blocks):
        if i%2 == 0:
            give_cue('word')
        elif i%2 == 1:
            give_cue('color')
        stims = random_stimuli(block)
        for word, color in stims:
            update_W(**update_kwargs)
            response, outcome = run_trial(word, color)
            if outcome == -1:
                update_W(**update_kwargs)
                response, outcome = run_trial(word, color)
    return Sim.outcome

def coeff_rate_sim(N):
    """ Run a bunch of simulations for varying coefficient rates.  Set the
        resolution with N, i.e. N = 20 will return a 20x20 array.
    """
    from collections import defaultdict
    from itertools import product
    block_lengths = random_blocks(8, 40, 30)
    outcome_rates = defaultdict(list)
    rates = np.linspace(0.0002, 0.002, N)
    for w_rate, c_rate in product(rates, rates):
        reset_sim()
        update_kwargs = {'word_rate':w_rate, 'color_rate':c_rate, 
                         'trial_interval':250,
                         'word_limits':(0.05,0.95), 'color_limits':(0.3,0.7)}
        run_simulation(block_lengths, **update_kwargs)
        for each in ['hits', 'errors', 'hesitations']:
            outcome_rates[each].append(outcome_proportion(each))
        
    arrays = { each:np.array(outcome_rates[each]).reshape(N,N) 
               for each in ['hits','errors', 'hesitations'] }
    return arrays
