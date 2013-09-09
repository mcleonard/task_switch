import matplotlib.pyplot as plt
import numpy as np

def session_plot(Sim):
	""" Create a plot of a simulated session. """

	fig, ax1 = plt.subplots(figsize=(9,6))

	outcomes = np.array(Sim.outcome)
	good = np.where(outcomes==1)[0]
	bad = np.where(outcomes==0)[0]
	pause = np.where(outcomes==-1)[0]

	ax1.scatter(good, [1]*len(good), marker='o', 
				facecolor='none', edgecolor='g', s=50)
	ax1.scatter(bad, [0]*len(bad), marker='x', c='r', s=50)
	ax1.scatter(pause, [-1]*len(pause), marker='d', c='grey', s=50)
	ax1.set_xlabel('Time step')
	ax1.set_ylabel('Outcome')
	ax1.set_yticks([-1, 0, 1])
	ax1.set_yticklabels(['Hesitation','Wrong', 'Correct'])
	ax1.set_ylim((-1.1, 1.1))

	colors = plt.cm.RdBu((50, 220))
	labels = [r'c_color', r'c_word']
	ax2 = ax1.twinx()
	lines = ax2.plot(Sim.W_coeffs)
	for line, color in zip(lines, colors):
	    line.set_color(color)
	ax2.set_ylabel('Coefficients')
	ax2.set_ylim((-0.2,1.2))
	ax2.set_xlim((-1,len(outcomes)))
	_ = ax2.legend((r"$\displaystyle c_{word}$", 
					r"$\displaystyle c_{color}$"), 
					loc='lower right', fontsize=18)

def outcome_contours(x, y, hits, errors, hesitations):
	""" Create some contour plots showing the proportion of outcomes. 
	
		x, y are the values of the x and y axes, respectively.

		hits, errors, hesitiations should be 2D arrays of values.
	"""
	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,6))
	outcomes = [hits, errors, hesitations]
	titles = ['Hits', 'Errors', 'Hesitations']
	zipped = zip(axes, outcomes, titles)
	ticklabelFormatter = matplotlib.ticker.ScalarFormatter()
	ticklabelFormatter.set_scientific(True)
	ticklabelFormatter.set_powerlimits((-2,3))
	for ax, outcome, title in zipped:
	    CS = ax.contourf(x, y, outcome, 9, cmap=plt.cm.gist_heat)
	    ax.set_aspect('equal')
	    ax.set_title(title)
	    ax.xaxis.set_major_formatter(ticklabelFormatter)
	    ax.yaxis.set_major_formatter(ticklabelFormatter)
	    cb = plt.colorbar(CS, ax=ax, shrink=0.4)
	    cb.ax.tick_params(length=0)
	fig.tight_layout()

	return axes