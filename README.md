# FFNSL: Feed-Forward Neural-Symbolic Learner
FFNSL experiments to support the paper "FFNSL: Feed-Forward Neural-Symbolic Learner". This code has been tested on MacOSX Big Sur 11.6.1.

# Dependencies
* python >= 3.7
* [ILASP](www.ilasp.com) including clingo.
* [FastLAS](www.github.com/marklaw/FastLAS-public)
* [Graphviz](https://graphviz.org/)

# Installation
From the project root:
1. `pip install -r requirements.txt`
2. `python setup.py`

# Data and Neural Network weights
Data and pre trained models are available to download [here](https://drive.google.com/file/d/1FZtAgnJsG89q7PSaDF-yHUBJ5tZcRNKC/view?usp=sharing). Extract the zip folder and copy the files into the 
directory tree.

# To run paper experiments
Refer to the `run_experiments.md` file in each example directory. 
Results presented in the FFNSL paper are available in the `paper_results` folder of each example.

