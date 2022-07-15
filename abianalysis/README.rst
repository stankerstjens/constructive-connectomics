abianalysis
===========

The cells of the brain are ultimately born from a single source cell.  We
investigate whether this single root ancestor, together with properties of cell
division, induces a systematic organization among the cells of a developing
brain that contributes to the its self-construction into a functional entity.

The code in this package helps the exploration of this hypothesis by
facilitating the

1. analysis of spatial gene expression data from the Allen Institute of
   Brain Science [ABI]_;
2. specification and simulation of models of cell growth and expression
   that encapsulate our hypotheses; and
3. visualization of analyses and simulations.


Example
-------

>>> from abianalysis import Volume, make_balanced_hierarchy
>>> volume = Volume.load('E11.5')
>>> volume.preprocess()
>>> hierarchy = make_balanced_hierarchy(volume, n_iterations=16)
>>> print(len(list(hierarchy.just_not_leaves())))
16
>>> volume = Volume.load('E11.5')
>>> volume.preprocess()
>>> hierarchy = make_balanced_hierarchy(volume, n_regions=16)
>>> print(len(list(hierarchy.just_not_leaves())))
16
>>> volume = Volume.load('E11.5')
>>> volume.preprocess()
>>> hierarchy = make_balanced_hierarchy(volume, n_regions=16)
>>> print(len(list(hierarchy.just_not_leaves())))
16

Features
--------

* Loading voxelated gene expression data from the Allen Brain Institute [ABI]_.
* Decomposing a volume of expression voxels into a hierarchy of nested regions.
* Transforming voxel grid of gene expression to a spatial graph.
* Performing navigation through spatial gene expression graphs.
* Simulating stochastic models of gene expression at cell division.
* Simulating the placement of cells in a growing tissue


Installation
------------

First install using pip. Then download the supporting data as follows::

    $ pip install abianalysis
    $ ... something to download data ...


License
-------

For now, all rights are reserved.  Probably, this project will published under
a creative commons license along with a scientific publication.

References
----------

.. [ABI] Allen Institute for Brain Science; developing mouse brain atlas
         https://developingmouse.brain-map.org

