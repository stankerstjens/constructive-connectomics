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

## Running the Demos

The demos are in the `demos/` folder and provide an interactive visualization of
some of the data.  The demos can be run in a browser.