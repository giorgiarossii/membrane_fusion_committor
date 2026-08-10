# PLUMED inputs for committor driven NP-mediated stalk formation

This directory contains the PLUMED inputs associated with:

"Let's Stalk About Membranes: Committor-Based Enhanced Sampling
of Stalk Formation"

## Biased simulation

The `biased/` directory contains:

- the PLUMED input;
- the required GROMACS index;
- the trained committor model;
- the custom PLUMED actions.

Run the PLUMED syntax test from inside `biased/`:

plumed driver --natoms 100000 --parse-only --kt 2.5 --plumed plumed.dat

The custom actions are compiled automatically through `LOAD FILE=*.cpp`.

The inputs were developed and tested with PLUMED 2.9.4 compiled with LibTorch support.