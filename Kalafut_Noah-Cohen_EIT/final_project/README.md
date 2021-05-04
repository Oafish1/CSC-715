# Solving EIT via PDE constrained optimization

This is a python implementation of a optimization-based estimation of the EIT problem, using PDE constrained optimization, without regularization. 

## Requirements: 

-numpy==1.16.0

-scipy==1.5.4

For some of the examples we will use the h5py library to extract the data (already contained in the tests folder)

## Files

-Kalafut_Noah-Cohen_EIT.pdf: A write-up of the problem and the solutions used within fem.py

-eit/fem.py: The main solver file.  Implements methods used to solve EIT problems including spaces, DtN mapping, adjoint-state solvers.

-setup.py: The configuration file for installation.

-test/test_*.py: If run, tests the specified aspect of the solver for fem.py.

## Installation

To install just go to the main directory and install as usual: 

python setup.py install

This will install the package on your local python environment (we strongly recommend to create a virtual environment before hand)

## Tests

We have added a context.py script in the test folder.  This allows the test files to be run without having to install the package on your local environment. 

## Made by

https://github.com/Forgotten/EIT