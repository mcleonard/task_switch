""" Hopfield network code tests """

import nose
import numpy as np
import hopfield as hf

def setup():
	net = hf.Hopfield()
	mock_state = np.array([-])

def test_creation():
	net = hf.Hopfield()
	assert type(net) == hf.Hopfield

@with_setup(setup)
def test_net_update():
	net.state = 

def test_net_run():
	pass

def test_net_eq():
	pass

def test_net_train():
	pass

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