# Networkqit package

The spectral entropy framework
------------------------------

An information theoretic approach inspired by quantum statistical mechanics was recently proposed as a means 
to optimize network models and to assess their likelihood against synthetic and real-world networks.
Importantly, this method does not rely on specific topological features or network descriptors, 
but leverages entropy-based measures of network distance.
Entertaining the analogy with thermodynamics, **networkqit** provides a computational tool for the estimate of 
spectral entropy and fitting of model parameters.
These results enable the practical application of this novel and powerful framework to network model inference.

Documentation
-------------
The full documentation for this package under development and is currently being written, its current version is available at:

https://networkqit.github.io/

Installation
------------

To use the **networkqit** package we suggest to use `virtualenv`.
The **networkqit** package is pure Python, so it should work on Linux, Mac OSX and Windows.
Here we report instructions for usage in a standard Ubuntu linux installation.

How to install on Linux
-----------------------

1. Open a terminal, install `pip` and `virtualenv` and clone this repository
	
	
    cd
	sudo apt-get install python3-pip
	sudo pip3 install virtualenv
	virtualenv workspace
	cd workspace
	git clone https://bitbucket.org/carlonicolini/networkqit
	
2. You cloned the repository. Now start the virtualenv session.


	source bin/activate

3. If you are inside the `virtualenv` session, check that the Python version you are using is the one provided by `virtualenv`.


	which python3

4. Now install the networkqit package within the `virtualenv` environment.


	cd networkqit
	python3 setup.py sdist

Now install the created Python package, that should come with all its dependencies `matplotlib`, `numpy`, `networkx`, `pandas`, `numdifftools`, `bctpy`


	cd ..
	pip3 install networkqit/dist/networkqit-0.1.tar.gz 
