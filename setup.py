##
## particlefever setup
##
from setuptools import setup

## Definition of the current version
VERSION = "0.1"

## Generate a __version__ attribute
## for the module
with open("./paper_metachange/__init__.py", "w") as version_out:
      version_out.write("__version__ = \"%s\"\n" %(VERSION))

long_description = open("README.md").read()

setup(name = 'paper_metachange',
      version = VERSION,
      description = "Code for manuscript on probabilistic adaptation in changing environments",
      long_description = long_description,
      author = 'Yarden Katz',
      author_email = 'yarden@hms.harvard.edu',
      maintainer = 'Yarden Katz',
      maintainer_email = 'yarden@hms.harvard.edu',
      packages = ['paper_metachange'],
      platforms = 'ALL',
      install_requires = ["numpy", "scipy", "matplotlib", "pandas",
                          "seaborn", "ruffus"],
      keywords = ['microbiology', 'evolution', 'systems biology',
                  'science', 'bayesian', 'inference', 
                  'markov-models', 'probabilistic-modeling',
                  'particle-filtering', 'monte-carlo'],
      classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        ]
      )

