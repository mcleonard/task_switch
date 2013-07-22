""" Module containing pre-defined states used in the simulation. """

import numpy as np
import hopfield

HIT = 1
ERROR = 0

# Defining states for the Hopfield network
colors = {'red':np.array([1,1,1,1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]),
		  'blue':np.array([-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1]),
		  'green':np.array([-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1])}

words = {'red':np.array([1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1]),
		 'blue':np.array([-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1, 1,-1,-1]),
		 'green':np.array([-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1])}

# Training Weight matrix for color outputs
net = hopfield.Hopfield()
for c in colors:
	net.train(colors[c])
W_ink = net.W

# Training Weight matrix for word outputs
net = hopfield.Hopfield()
for w in words:
	net.train(words[w])
W_wrd = net.W