# Networkqit package

## Experimental library for manipulation of network models within the spectral entropy framework

# Documentation
The full documentation for this package is being written, its current version is available at:


## Installation

To use the networkqit package we suggest to use `virtualenv`. The `networkqit` package is pure Python, so it should work on Linux, Mac OSX and Windows. Here we report instructions for usage in a standard Ubuntu linux installation.

## How to install on Linux
### 1. Open a terminal, install `pip` and `virtualenv` and clone this repository
	
	cd
	sudo apt-get install python3-pip
	sudo pip3 install virtualenv
	virtualenv workspace
	cd workspace
	git clone https://bitbucket.org/carlonicolini/networkqit
	
### 2. You cloned the repository. Now start the virtualenv session.

	source bin/activate

### 3. If you are inside the `virtualenv` session, check that the Python version you are using is the one provided by `virtualenv`.

	which python3

### 4. Now install the networkqit package within the `virtualenv` environment.

	cd networkqit
	python3 setup.py sdist

Now install the created Python package, that should come with all its dependencies `matplotlib`, `numpy`, `networkx`, `pandas`, `numdifftools`, `bctpy`

	cd ..
	pip3 install networkqit/dist/networkqit-0.1.tar.gz 


## Examples:
