# surfStats

This is a collection of notebooks and functions aimed at mapping the statistics of land surfaces.  

## Installation

GDAL must already be available in the environment before installing — its Python bindings need
to match the system `libgdal`, so it isn't pip-installed automatically. The conda-forge
environment in `environment.yml` provides it (`conda env create -f environment.yml`).

With that in place:

```
pip install -e .
```
