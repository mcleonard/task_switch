""" Module for running the task switching simulation. """

import numpy as np

import constants
import hopfield

color_change_coef = .03
word_change_coef = .06
coeff_upper_limit = 1
coeff_lower_limit = 0

class Task(object):
	""" This object is going to hold the important task variables and 
		simulated results.
	"""
	actual_task = 'word'
	assumed_task = 'word'
	stimulus = []
	response = []
	outcome = []
	network = hopfield.Hopfield()
	W_coeff = [.5, .5] # Color, Word
	tau = 1

# We want to use the cue to indicate which W matrix to use in our Hopfield
# network.  If we aren't given a cue, we'll set the W matrices equally.  For 
# each trial, we'll take an input stimulus and get the output from the 
# network.  If there is no cue, we'll update our W matrices based on the
# outcome.

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
		Task.W_coeff[0] += color_change_coef
		Task.W_coeff[1] -= word_change_coef
	elif task == 'word':
		Task.W_coeff[0] -= color_change_coef
		Task.W_coeff[1] += word_change_coef

	# Make sure the coefficients are inside limits
	for i, c in enumerate(Task.W_coeff):
		if c > coeff_upper_limit:
			Task.W_coeff[i] = coeff_upper_limit
		elif c < coeff_lower_limit:
			Task.W_coeff[i] = coeff_lower_limit

	W_full = (Task.W_coeff[0]*constants.W_ink + 
			  Task.W_coeff[1]*constants.W_wrd)
	Task.network.W = W_full

def random_stimuli(size=1):
	""" Generate random word/color pairs. """
	colors = ['red', 'blue', 'green']
	pairs = np.random.choice(colors, (2, size)).T 
	return pairs

def give_cue(cue: ('word', 'color')):
	""" Cues the model to one task. """
	Task.assumed_task = cue

def run_trial(word, color):
	""" Simulate one trial given the word and color. """
	
	network = Task.network
	stimulus = gen_stimulus(word, color)

	# Keeping track of the task
	Task.stimulus.append(stimulus)

	# First we'll update our weights based on our assumed task
	update_W(Task.assumed_task)
	
	# Now get the response from the stimulus
	network.state = stimulus
	# Turns out you can fine tune the model with the number of iterations
	# and the initial temperature.  You can basically get it to do really
	# well or really poorly by adjusting these parameters.
	network.run(200, init_temp=0.1)
	response = network.state
	Task.response.append(response)

	# Now see if the response was correct or not
	if Task.actual_task == 'color':
		outcome = (response == constants.colors[color]).all()
	elif Task.actual_task == 'word':
		outcome = (response == constants.words[word]).all()
	Task.outcome.append(outcome)



