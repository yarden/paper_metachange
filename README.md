Probabilistic adaptation in changing microbial environments
===========================================================

Code for the manuscript:

Yarden Katz and Michael Springer, "Probabilistic adaptation in changing microbial environments", July 2016.

Installing the code
-----------------

The code can be installed as a regular Python package using ``pip install .`` in the repository directory (or ``python setup.py install``). Unit tests can be run using: 

    cd ./paper_metachange
    python testing.py

Organization
-----------------

The directories are organized as follows:

1. ``paper_metachange``: Python module containing code used in paper
2. ``data``: growth rate data 
3. ``sbml_models``: biochemical model from paper in SBML (.xml) format
4. ``simulations_params``: parameters used for simulations
5. ``nonpython_figures``: figures generated outside of Python

Producing figures
------------------

Figures are generated using a [ruffus](http://www.ruffus.org.uk/) pipeline by running:

    cd paper_metachange
    python make_paper.py

Particle filtering inference for Bayesian models is done using the [ParticleFever library](https://github.com/yarden/particlefever).
