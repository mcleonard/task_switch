""" Module for contructing, training, and using Hopfield networks. """

import numpy as np
import matplotlib.pyplot as plt

class Hopfield(object):

	def __init__(self, state=None, W=None):
		self.W = W	# Weight matrix
		self.state = state
		self.trained_states = 0
		self.shape = None

	def run(self, n_iters=30, init_temp=1):
		""" Run the Hopfield network using simulated annealing. """
		
		for t in range(n_iters):
			curr_temp = 0.15 + init_temp*np.exp(-t/5.)
			self.update(tau=curr_temp)

		return self.state

	def update(self, tau=0.1):
		""" Update all the units once. """
		
		# First part of the training rule, take the sums
		sums = np.dot(self.W, self.state)
		ps = sigmoid(sums/tau)
		# Then take the threshold and change to 1, -1 from 1, 0
		new_state = (np.random.rand(len(self)) < ps)*2 - 1
		self.state = new_state
		return self.state

	def train(self, Y):
		""" Train the weight matrix W on input Y. """
		
		n = self.trained_states

		# This performs the actual Hebbian learning
		new_W = np.array([ y.repeat(len(Y))*Y for y in Y])
		
		if self.W is None:
			self.W = new_W
		else:
			old_W = self.W*n
			self.W = 1/float(n+1) * (old_W + new_W)
		
		self.trained_states += 1
		
		# Make sure diagonal is 0
		self.W = self.W - np.diag(self.W.diagonal())

	def plot(self, shape=None, fig_num=None):
		if shape is not None:
			pass
		elif self.shape is not None:
			shape = self.shape
		
		plot_state(self.state, shape=shape, fig_num=fig_num)

	def __len__(self):
		return len(self.state)

	def __eq__(self, other):
		compare = self.state == other

		if compare.all():
			return True
		elif (~compare).all():
			print("Negation.")
			return False
		else:
			return False

	def __repr__(self):
		return self.state.__repr__()

def random_state(N=100):
	""" Generate a random state with N units. """
	return np.random.randint(0, 2, N)*2-1

def flip_units(X, N=1):
	""" Flip N random units in the state X, not in place."""
	indices = np.random.choice(range(len(X)), size=N, replace=False)
	Y = X.copy()
	Y[indices] = -X[indices]
	return Y

def sigmoid(x):
	y = 1/(1 + np.exp(-x))
	return y

def plot_state(state, shape=None, fig_num=None):
	""" Plots the given state as an image of squares indicating the state of 
		each unit. 
	"""

	if fig_num:
		fig = plt.figure(fig_num)
		fig.clf()
	else:
		fig = plt.figure()
	ax = fig.add_subplot(111)

	if shape is None:
		shape = (1, len(state))

	s = state.reshape(shape)
	_ = ax.imshow(s, interpolation='nearest', cmap=plt.cm.gray)
	_ = ax.set_xticks(np.arange(-0.5, -0.5+shape[1]))
	_ = ax.set_yticks(np.arange(-0.5, -0.5+shape[0]))
	_ = ax.set_xticklabels('')
	_ = ax.set_yticklabels('')
	_ = ax.grid(b=True, which='major', color='white', ls='-')
	ax.tick_params(length=0)