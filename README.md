# Constructive Connectomics

Repository accompanying the paper 'Constructive Connectomics: how neuronal axons get from here to there using their family tree


## Installation

First clone the repository. The software was last tested using python version 3.10.3. We installed python on MacOS using pyenv:

```bash
$ pyenv install 3.10.3
$ pyenv virtualenv 3.10.3 constructive-connectomics
$ pyenv local constructive-connectomics
```

The packages and notebook extensions were installed using

```bash
$ pip install Cython
$ pip install ./abianalysis
$ pip install ./pylineage
$ pip install jupyter pythreejs
$ jupyter nbextension install --py pythreejs
$ jupyter nbextension enable --py
```


## Running the notebooks

Start the jupyter notebook server by running

```bash
$ jupyter notebook
```

This will open a web-page. Navigate to the `notebooks/` folder. Note that some 
notebooks might take a long time to run (order of hours, not days), or fail if 
the memory capacity of the machine is insufficient. (All notebooks were run 
successfully on a MacBook Pro with an M1 chip and 64GB of memory.)  

## Example

A minimal example for how to run a simulation of 100000 cells expression 200 
genes, and embedded in 3D space; and collect the expression and positions of the 
leaves in two lists:

```python
from pylineage.multi_lineage_simulator import MultiLineageSimulator

sim = MultiLineageSimulator(n_dims=3, n_divisions=100000 - 100, n_roots=100,
                            n_genes=200, symmetric_prob=0.2)
sim.run()

original_root = list(sim.root.children)[0]
leaves = list(original_root.leaves())

leaf_exp = [leaf.expression for leaf in leaves]
leaf_pos = [leaf.expression for leaf in leaves]
```
