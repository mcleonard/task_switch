""" Module for running the task switching simulation. """

import numpy as np

import constants
import hopfield

color_change_coef = .03
word_change_coef = .06
coeff_upper_limit = 1
coeff_lower_limit = 0

class Sim(object):
	""" This object is going to hold the important task variables and 
		simulated results.
	"""
	actual_task = 'word'
	assumed_task = 'word'
	stimulus = []
	response = []
	outcome = []
	network = hopfield.Hopfield()
	tau = 1

def reset_sim():
	""" Reset the simulation. """
	Sim.actual_task = 'word'
	Sim.assumed_task = 'word'
	Sim.stimulus = []
	Sim.response = []
	Sim.outcome = []
	Sim.network = hopfield.Hopfield()
	Sim.W_coeff = np.array([.5, .5]) # Color, Word
	Sim.tau = 1

def gen_stimulus(word, color):
	""" Generate the stimulus state for word and color """
	index = np.where((constants.words[word] + 
			  		 constants.colors[color]) > 0 )
	state = -np.ones(16)
	state[index] = 1
	return state

def update_W(task: ('word', 'color') = None):
	""" Update W matrix for the response network based on the cue. """
	if task == 'color':
		Sim.W_coeff[0] += color_change_coef
		Sim.W_coeff[1] -= word_change_coef
	elif Sim == 'word':
		Sim.W_coeff[0] -= color_change_coef
		Sim.W_coeff[1] += word_change_coef

	# Make sure the coefficients are inside limits
	for i, c in enumerate(Sim.W_coeff):
		if c > coeff_upper_limit:
			Sim.W_coeff[i] = coeff_upper_limit
		elif c < coeff_lower_limit:
			Sim.W_coeff[i] = coeff_lower_limit

	W_full = (Sim.W_coeff[0]*constants.W_ink + 
			  Sim.W_coeff[1]*constants.W_wrd)
	Sim.network.W = W_full

def random_stimuli(size=1):
	""" Generate random word/color pairs. """
	colors = ['red', 'blue', 'green']
	pairs = np.random.choice(colors, (2, size)).T 
	return pairs

def give_cue(cue: ('word', 'color')):
	""" Cues the model to one task. """
	Sim.assumed_task = cue

def run_trial(word, color, net_iters=200, net_temp=1):
	""" Simulate one trial given the word and color. """
	
	network = Sim.network
	stimulus = gen_stimulus(word, color)

	# Keeping track of the stimulus
	Sim.stimulus.append(stimulus)
	
	# Now get the response from the stimulus
	network.state = stimulus
	# Turns out you can fine tune the model with the number of iterations
	# and the initial temperature.  You can basically get it to do really
	# well or really poorly by adjusting these parameters.
	network.run(net_iters, init_temp=net_temp)
	response = network.state
	Sim.response.append(response)

	# Now see if the response was correct or not
	if Sim.actual_task == 'color':
		outcome = (response == constants.colors[color]).all()
	elif Sim.actual_task == 'word':
		outcome = (response == constants.words[word]).all()
	Sim.outcome.append(outcome)

#****************************************************************************#

# Initialize the response network's weight matrix
W_full = (Sim.W_coeff[0]*constants.W_ink + 
		  Sim.W_coeff[1]*constants.W_wrd)
Sim.network.W = W_full

