# efficient_graph_algorithms


This is the PyTorch implementation of ["Efficient Graph Field Integrators Meet Point Clouds"](https://arxiv.org/abs/2302.00942). 

Krzysztof Choromanski\*, Arijit Sehanobish\*, Han Lin\*, Yunfan Zhao\*, Eli Berger, Tetiana Parshakova, Alvin Pan, David Watkins, Tianyi Zhang, Valerii Likhosherstov, Somnath Basu Roy Chowdhury, Avinava Dubey, Deepali Jain, Tamas Sarlos, Snigdha Chaturvedi, Adrian Weller


Google Research, Columbia University, Haifa University, Stanford University, The Boston Dynamics AI Institute, University of Cambridge, The University of North Carolina at Chapel Hill, The Alan Turing Institute.

The Fortieth International Conference on Machine Learning (ICML), 2023


## Installation
```bash
git clone git@github.com:Topographers/efficient_graph_algorithms.git
cd efficient_graph_algorithms
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
pip3 install -e . --user
git clone https://github.com/topographers/planar_separators.git
```
If you have error when running command line ```pip3 install -e . --user```, you can follow this [link](https://github.com/microsoft/vscode-python/issues/14327#issuecomment-757408341).


## Coding Guidelines

We have a strict coding guideline that is written [here](docs/coding_guidelines.md). This should be read thoroughly before contribution to this repository.

## Experiments
For information on how to run each experiment:

* [scripts/experiments/gaussian_kernel_test.py](docs/experiments/gaussian_kernel_test.md)
* [scripts/experiments/meshgraphdata_interpolator_3dplot.py](docs/experiments/meshgraphdata_interpolator_3dplot.md)
* [scripts/experiments/placebo_separator_test.py](docs/experiments/placebo_separator_test.md)
* [scripts/experiments/separator_test.py](docs/experiments/separator_test.md)
* [scripts/experiments/graph_diffusion_gf_integrator_test.py](docs/experiments/graph_diffusion_integrator_test.md)

## MeshGraphNet Datasets
For information on how to download and prepare meshgraphnet dataset:

* [docs/experiments/prepare_graph_mesh_data.md](docs/experiments/prepare_graph_mesh_data.md)
