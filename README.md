# efficient_graph_algorithms
Implementations of efficient graph algorithms

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

## Coding Guidelines

We have a strict coding guideline that is written [here](docs/coding_guidelines.md). This should be read thoroughly before contribution to this repository.

## Experiments
For information on how to run each experiment:

* [scripts/experiments/gaussian_kernel_test.py](docs/experiments/gaussian_kernel_test.md)
* [scripts/experiments/interpolation_test.py](docs/experiments/interpolator_test.md)
* [scripts/experiments/meshgraphdata_interpolator_3dplot.py](docs/experiments/meshgraphdata_interpolator_3dplot.md)
* [scripts/experiments/placebo_separator_test.py](docs/experiments/placebo_separator_test.md)
* [scripts/experiments/separator_test.py](docs/experiments/separator_test.md)
* [scripts/experiments/graph_diffusion_gf_integrator_test.py](docs/experiments/graph_diffusion_integrator_test.md)

## MeshGraphNet Datasets
For information on how to download and prepare meshgraphnet dataset:

* [docs/experiments/prepare_graph_mesh_data.md](docs/experiments/prepare_graph_mesh_data.md)
