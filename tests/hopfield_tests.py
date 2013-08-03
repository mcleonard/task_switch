""" Hopfield network code tests """

import nose
import numpy as np
import hopfield as hf

def setup():
	net = hf.Hopfield()
	mock_state = np.array([1,1,-1,1,1,-1,-1,1,-1,-1])

def setup_trained_net():
	net = hf.Hopfield()
	mock_state = np.array([1,1,-1,1,1,-1,-1,1,-1,-1])
	net.train(mock_state)

def test_creation():
	net = hf.Hopfield()
	assert type(net) == hf.Hopfield

@with_setup(setup_trained_net)
def test_net_update():
	net.update()

def test_net_run():
	net.state = mock_state


def test_net_eq():
	pass

def test_net_train():
	try:
		net.train(mock_state)

def test_net_plot():
	try:
		_ = net.plot()
	except:
		raise

def test_random_state():
	pass

def test_flip_units():
	pass

def test_sigmoid():
	pass

def test_plot_state():
	pass