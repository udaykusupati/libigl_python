# libigl_python
Testing features from Libigl bound to python.

## Install environment

Run the following commands in your terminal to create and activate the conda environment:

```
conda create -y --name gcm_env python=3.7
conda env create -f environment.yml
conda activate gcm_env
```

Then you need to install a recent version of vtk (9.0.3), PyQt5 (5.15.4), then Mayavi (4.7.3).

```
pip install vtk==9.0.3
pip install PyQt5==5.15.4
pip install mayavi==4.7.3
```

Other versions might be compatible, but we advise to use those.

## Run jupyter notebook

Use jupyter notebook to use libigl, for some reason I had issues with jupyter lab.

```
jupyter notebook
```