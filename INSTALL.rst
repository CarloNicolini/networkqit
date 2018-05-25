Install
=======

networkqit requires Python 2.7, 3.4, 3.5, or 3.6.  If you do not already
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
your computer and you intend to install ``networkqit`` inside of it.  If you want
to create and work with Python virtual environments, please follow instructions
on `venv <https://docs.python.org/3/library/venv.html>`_ and `virtual
environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

First, make sure you have the latest version of ``pip`` (the Python package manager)
installed. If you do not, refer to the `Pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip`` first.

Install the released version
----------------------------

Install the current release of ``networkqit`` with ``pip``::

    $ pip install networkqit

To upgrade to a newer release use the ``--upgrade`` flag::

    $ pip install --upgrade networkqit

If you do not have permission to install software systemwide, you can
install into your user directory using the ``--user`` flag::

    $ pip install --user networkqit

Alternatively, you can manually download ``networkqit`` from
`GitHub <https://bitbucket.org/carlonicolini/networkqit/>`_  or
`PyPI <https://pypi.python.org/pypi/networkqit>`_.
To install one of these versions, unpack it and run the following from the
top-level source directory using the Terminal::

    $ pip install .

Install the development version
-------------------------------

If you have `Git <https://git-scm.com/>`_ installed on your system, it is also
possible to install the development version of ``networkqit``.

Before installing the development version, you may need to uninstall the
standard version of ``networkqit`` using ``pip``::

    $ pip uninstall networkqit

Then do::

    $ git clone https://bitbucket.org/carlonicolini/networkqit
    $ cd networkqit
    $ pip install -e .

The ``pip install -e .`` command allows you to follow the development branch as
it changes by creating links in the right places and installing the command
line scripts to the appropriate locations.

Then, if you want to update ``networkqit`` at any time, in the same directory do::

    $ git pull

Optional packages
-----------------

.. note::
   Some optional packages (e.g., `scipy`, `gdal`) may require compiling
   C or C++ code.  If you have difficulty installing these packages
   with `pip`, please review the instructions for installing
   the full `scientific Python stack <https://scipy.org/install.html>`_.

The following optional packages provide additional functionality.

- `pandas <http://pandas.pydata.org/>`_ (>= 0.20.0) provides a DataFrame, which
  is a tabular data structure with labeled axes.
- `Matplotlib <http://matplotlib.org/>`_ (>= 2.0.2) provides flexible drawing of
  graphs.
- `drawnow <https://pypi.org/project/drawnow/>`_(>=0.72.0) is a MATLAB-like drawnow command to monitor the optimization process.
- `numdifftools <https://pypi.org/project/Numdifftools/>`_ (>=0.9.20) is a suite of tools written in _Python to solve automatic numerical differentiation problems in one or more variables. It can be necessary to exactly compute gradients of relative entropy.

- `seaborn <https://pypi.org/project/seaborn/>`_ (>=0.8.1) is a library for making attractive and informative statistical graphics in Python.

To install ``networkqit`` and all optional packages, do::

    $ pip install networkqit[all]

To explicitly install all optional packages, do::

    $ pip install numpy scipy pandas matplotlib seaborn numdifftools drawnow

Or, install any optional package (e.g., ``numpy``) individually::

    $ pip install numpy

Installation in the virtualenv environment
------------------------------------------

We suggest to use a Python3 virtualenv to setup the networkqit package.
