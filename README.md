# Committor-based enhanced sampling of NP-mediated stalk formation

Supporting material for 'Paper Title' by Giorgia Rossi, Enrico Trizio, Davide Bochicchio, Giulia Rossi, and Michele Parrinello.

## Contents

- `biased/`  
  COLVAR files generated during the iterative biased simulations.

  - `biased/template/`  
    Files required to reproduce the biased simulations, including:
    1. code for committor-based biasing (pytorch_model_bias.cpp)
    3. code for fast calculation of shell-coordination descriptors (coordination_multi.cpp)
    4. adapted chain-coordinate collective variable (MemFusionP_NP.cpp)

- `unbiased/`
  Data obtained from unbiased simulations initiated from the metastable basins.

- `models/`  
  Trained committor models used during the different iterations.

- `training_example.ipynb`  
  Example notebook showing how to train the committor model.

## NOTE 
For an updated version of the code and tutorials, also on other machine learning collective variables, check our [mlcolvar library](https://github.com/luigibonati/mlcolvar)
