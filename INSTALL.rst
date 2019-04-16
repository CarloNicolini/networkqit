Install
=======

networkqit requires Python 3.5 or newer.  If you do not already
have a Python environment configured on your computer, please see the
instructions for installing the full `scientific Python stack
<https://scipy.org/install.html>`_.

.. note::
   If you are on Windows and want to install optional packages (e.g., `scipy`),
   then you will need to install a Python distribution such as
   `Anaconda <https://www.anaconda.com/download/>`_,
   `Enthought Canopy <https://www.enthought.com/product/canopy>`_,
   `Python(x,y) <http://python-xy.github.io/>`_,
   `WinPython <https://winpython.github.io/>`_, or
   `Pyzo <http://www.pyzo.org/>`_.
   If you use one of these Python distribution, please refer to their online
   documentation.

Below we assume you have the default Python environment already configured on
your computer and you intend to install `**networkqit**` inside of it.  If you want
to create and work with Python virtual environments, please follow instructions
on `venv <https://docs.python.org/3/library/venv.html>`_ and `virtual
environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

First, make sure you have the latest version of ``pip`` (the Python package manager)
installed. If you do not, refer to the `Pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip`` first.

Installation from the Pypi repositories
---------------------------------------

**networkqit** has reached a pretty stable usability and is hosted on the Pypi repositories.
We currently only support Linux, here the detailed commands for installation on Ubuntu are given.
To install **networkqit** you just need to install `pip` at the latest Python3 version.

Ubuntu
~~~~~~

::

     sudo apt-get install python3-pip
     sudo pip3 install -U pip
     sudo pip3 install networkqit



Install the development version
-------------------------------

At the moment **networkqit** is in alpha stage and its API could change very fast.
For this reason **networkqit** is still not available on `pip`.
However you can manually download **networkqit** from the repository:
`Bitbucket <https://github.com/carlonicolini/networkqit/>`_ 

If you have `Git <https://git-scm.com/>`_ installed on your system, do

::

    git clone https://github.com/carlonicolini/networkqit
    cd networkqit
    pip install -e .

The ``pip install -e .`` command allows you to follow the development branch as
it changes by creating links in the right places and installing the command
line scripts to the appropriate locations. 
The package comes with its dependencies which get installed automatically.

Then, if you want to update **networkqit** at any time, in the same directory do::

    git pull

Optional packages
-----------------

.. note::
   Some optional packages (e.g., `scipy`, `gdal`) may require compiling
   C or C++ code.  If you have difficulty installing these packages
   with `pip`, please review the instructions for installing
   the full `scientific Python stack <https://scipy.org/install.html>`_.

The following optional packages provide additional functionalities, and are not necessary for the purely numerical calculations of spectral entropies.

- `pandas <http://pandas.pydata.org/>`_ (>= 0.20.0) provides a DataFrame, which
  is a tabular data structure with labeled axes.
- `Matplotlib <http://matplotlib.org/>`_ (>= 2.0.2) provides flexible drawing of
  graphs.
- `drawnow <https://pypi.org/project/drawnow/>`_ (>=0.72.0) is a MATLAB-like drawnow command to monitor the optimization process.
- `numdifftools <https://pypi.org/project/Numdifftools/>`_ (>=0.9.20) is a suite of tools written in _Python to solve automatic numerical differentiation problems in one or more variables. It can be necessary to exactly compute gradients of relative entropy.
- `seaborn <https://pypi.org/project/seaborn/>`_ (>=0.8.1) is a library for making attractive and informative statistical graphics in Python.

To install **networkqit** and all optional packages, do::

    pip install networkqit[all]

To explicitly install all optional packages, do::

    pip install numpy autograd scipy pandas matplotlib seaborn numdifftools drawnow

Or, install any optional package (e.g., ``numpy``) individually::

    pip install numpy

