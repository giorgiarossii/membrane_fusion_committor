# Committor-based enhanced sampling of NP-mediated stalk formation

Supporting material for ["Let's Stalk About Membranes: Committor-Based Enhanced Sampling of Stalk Formation"](https://arxiv.org/abs/2607.20122) by Giorgia Rossi, Enrico Trizio, Davide Bochicchio, Giulia Rossi, and Michele Parrinello.

## Contents

- `data/`
  - `biased/`: COLVAR files generated during the iterative biased simulations.
  - `unbiased/`: data obtained from unbiased simulations initiated from the metastable basins.
  - `models/`: trained committor models used during the different iterations.
  - `training/training_example.ipynb`: example notebook showing how to train the committor model.

- `plumed_inputs/`
  - PLUMED input and supporting files prepared for PLUMED-NEST.
  - Custom code for committor-based biasing (`pytorch_model_bias.cpp`).
  - Custom code for the shell-coordination descriptors (`coordination_multi.cpp`).
  - Adapted membrane-fusion collective variable (`MemFusionP_NP.cpp`).
  - Trained model, index, and simulation input files.

## PLUMED input

The input was developed and tested with PLUMED 2.9.4 compiled with LibTorch support. The custom actions are compiled automatically using `LOAD FILE=*.cpp`.

To test the input:

```bash
cd plumed_inputs/biased
plumed driver --natoms 100000 --parse-only --kt 2.49 --plumed plumed.dat
```

## Note

For updated code and tutorials on machine-learning collective variables, see the [mlcolvar library](https://github.com/luigibonati/mlcolvar).
