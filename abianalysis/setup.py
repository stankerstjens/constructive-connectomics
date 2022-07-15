"""Setup abianalysis"""
import setuptools

setuptools.setup(
    name="abianalysis",
    version="1.0.0",
    author="Stan Kerstjens",
    author_email="skerst@ini.ethz.ch",
    description="Package for analysis of ABI data",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "tqdm",
        "numpy",
        "scipy",
        "sklearn",
        "joblib",
        "requests",
        "igraph",
        "hsluv",
        "matplotlib"
    ],
)
