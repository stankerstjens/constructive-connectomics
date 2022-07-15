"""Setup for python code."""
import numpy
from Cython.Build import cythonize
from setuptools import find_packages, setup

setup(
    name="pylineage",
    version="0.1",
    author="Stan Kerstjens",
    author_email="skerst@ini.ethz.ch",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=['numpy', 'scipy', 'Cython'],
    ext_modules=cythonize(["pylineage/grid/ray.pyx",
                           "pylineage/grid/grid.pyx",
                           "pylineage/grid/cell_positioner.pyx"]),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
