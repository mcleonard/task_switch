""" Module for running task switching simulations. """

import numpy as np

import model

class records(object):
    """ This object is going to hold the simulated results.
    """
    stimulus = []
    response = []
    outcome = []
    coeffs =  [] # (Word, Color) tuples

def reset_sim():
    """ Reset the simulation. """
    records.stimulus = []
    records.response = []
    records.outcome = []
    records.coeffs =  [] # (Word, Color) tuples

def run_trial(word, color, net_iters=30, net_temp=0.5, **update_params):
    """ Simulate one trial given the word and color. """
    
    network = model.network
    stimulus = model.gen_stimulus(word, color)

    # Keeping track of the stimulus
    records.stimulus.append((word, color))
    
    # Now get the response from the stimulus
    network.state = stimulus
    network.run(n_iters=net_iters, init_temp=net_temp)
    
    response = network.state
    records.response.append(response)
    
    outcome = model.response_outcome(word, color, response)
    records.outcome.append(outcome)
    
    return response, outcome

def run_simulation(blocks, **update_params):
    """ Run a simulation for the block lengths in blocks. 

        Arguments
        ---------
        blocks : list of integers, lengths of blocks.
        update_params : parameters passed to update_W.
    """
    for i, block in enumerate(blocks):
        if i%2 == 0:
            model.give_cue('word')
        elif i%2 == 1:
            model.give_cue('color')
        stims = model.random_stimuli(block)
        for word, color in stims:    
            coeffs = model.update_W(**update_params)
            records.coeffs.append(coeffs)
            response, outcome = run_trial(word, color)
            if outcome == -1:
                coeffs = model.update_W(**update_params)
                records.coeffs.append(coeffs)
                response, outcome = run_trial(word, color)
    return records.outcome

def coeff_rates(N, rate_lims, blocks, **update_params):
    """ Run a bunch of simulations for varying coefficient rates.  Set the
        resolution with N, i.e. N = 20 will return a 20x20 array.

        Arguments
        ---------
        N : number of coefficient rates.
        rate_lims : tuple : lower and upper rate limits to vary over.
        blocks : list of integers, lengths of blocks.
        update_params : parameters passed to update_W.
    """
    from collections import defaultdict
    from itertools import product
    
    outcome_rates = defaultdict(list)
    rates = np.linspace(rate_lims[0], rate_lims[1], N)
    for w_rate, c_rate in product(rates, rates):
        reset_sim()
        update_params.update({'word_rate':w_rate, 'color_rate':c_rate})
        run_simulation(blocks, **update_params)
        for each in ['hits', 'errors', 'hesitations']:
            outcome_rates[each].append(model.outcome_proportion(each))
        
    arrays = { each:np.array(outcome_rates[each]).reshape(N,N) 
               for each in ['hits', 'errors', 'hesitations'] }
    return arrays

def incongruent_effect(task, N, interval=500, net_iter=30, net_temp=0.5):
    """ Simulate reaction times for non-switching blocks N trials long,
        for neutral, incongruent, and congruent stimuli.  
    """

    coeffs = {'word':(0.85, 0.15), 'color':(0.2, 0.7)}

    if task == 'color':
        neutral_stim = 'word'
    elif task == 'word':
        neutral_stim = 'color'

    neutral = model.random_stimuli(size=N, neutral=neutral_stim)
    congruent = model.random_stimuli(size=N, congruent=1)
    incongruent = model.random_stimuli(size=N)

    hesitations = []
    stims = [neutral, incongruent, congruent]
    for stim_set in stims:
        reset_sim()
        model.give_cue(task)
        net = model.network
        net.W = model.calculate_weights(coeffs[task])
        for word, color in stim_set:
            response, outcome = run_trial(word, color, net_iter=net_iter, net_temp=net_temp)
            if outcome == -1:
                response, outcome = run_trial(word, color, net_iter=net_iter, net_temp=net_temp)
        hes_rate = model.outcome_proportion('hesitations', records.outcome)
        hesitations.append(hes_rate)
    reaction_times = np.array([ model.reaction_time(interval, h) 
                                for h in hesitations ])
    return reaction_times
