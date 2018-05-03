# Networkqit package

## Experimental library for manipulation of network models within the spectral entropy framework

# Dependencies:

## Basic dependencies

	sudo pip3 install matplotlib numpy pandas networkx numdifftools
	
## Additional stuff

	sudo pip3 install cairocffi
	sudo pip3 install drawnow
	sudo pip3 install seaborn

	sudo apt-get install libffi-dev
	sudo pip3 install cairocffi


## Installation

To use the networkqit package we suggest to use `virtualenv`. The `networkqit` package is pure Python, so it should work on Linux, Mac OSX and Windows. Here we report instructions for usage in a standard Ubuntu linux installation.

### 1. Open a terminal, install `pip` and `virtualenv` and clone this repository
	
	cd
	sudo apt-get install python3-pip
	sudo pip3 install virtualenv
	virtualenv workspace
	cd workspace
	git clone https://bitbucket.org/carlonicolini/networkqit
	
### 2.. You cloned the repository. Now start the virtualenv session.

	source bin/activate

### 3. If you are inside the `virtualenv` session, check that the Python version you are using is the one provided by `virtualenv`.

	which python3

### 4. Now install the networkqit package within the `virtualenv` environment.

	cd networkqit
	python3 setup.py sdist

Now install the created Python package, that should come with all its dependencies `matplotlib`, `numpy`, `networkx`, `pandas`, `numdifftools`, `bctpy`

	cd ..
	pip3 install networkqit/dist/networkqit-0.1.tar.gz 


### 5. You can try to run this example of fitting the undirected binary configuration to the **karate club graph**, both using the standard maximum likelihood method or using the spectral entropy method:


	import numpy as np
	import pandas as pd
	import networkqit as nq
	import networkx as nx
	from networkqit.graphtheory.models.GraphModel import UBCM

	# 1. Generate the adjacency matrix of the karate club graph
	# In the networkqit framework we always treat graphs as adjacency matrices expressed as numpy arrays
	# do not use numpy matrix
	A = nx.to_numpy_array(nx.karate_club_graph())

	
	# 2. Create the standard Maximum Likelihood solver, for a problem with N variables
	# randomly initialized
	solver = nq.MLEOptimizer(A=A, x0=np.random.random((len(A),1)))
	# call the setup function to create the solver's internals
	solver.setup() 
	# Run the optimization
	sol = solver.runfsolve(model='UBCM')

## Examples:
