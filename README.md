# Efficient Graph Field Integrators

Given weighted undirected graph $G=(V, E, W)$, a kernel $K:V\times V \rightarrow \mathbb{R}$ 
and a tensor field $\mathcal{F}:V \rightarrow \mathbb{R}^{d_{1} \times \ldots \times d_{l}}$ 
defined on $V$, where $d_{1},\ldots,d_{l}$ stand for tensor dimensions.


In this repository, we implement several methods that allow efficient computation of
```math
i(v) := \sum_{w \in \mathrm{V}}\mathrm{K}(w,v)\mathcal{F}(w), \qquad \text{for all } v \in V.
```
We refer to the process of computing $i(v)$ as **graph-field integration** (GFI).


This repository accompanies the paper ["Efficient Graph Field Integrators Meet Point Clouds"](https://arxiv.org/abs/2302.00942). 

Krzysztof Choromanski\*, Arijit Sehanobish\*, Han Lin\*, Yunfan Zhao\*, Eli Berger, Tetiana Parshakova, Alvin Pan, David Watkins, Tianyi Zhang, Valerii Likhosherstov, Somnath Basu Roy Chowdhury, Avinava Dubey, Deepali Jain, Tamas Sarlos, Snigdha Chaturvedi, Adrian Weller


Google Research, Columbia University, Haifa University, Stanford University, The Boston Dynamics AI Institute, University of Cambridge, The University of North Carolina at Chapel Hill, The Alan Turing Institute.

The Fortieth International Conference on Machine Learning (ICML), 2023

<p align="center">
<img src="https://github.com/topographers/efficient_graph_algorithms/blob/main/image.png?raw=true"  width="400px"/>
</p>

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

## Getting started

This repository contains implementations of several GFIs that inherit from `GFIntegrator`.
They can be categorized based on their representation of point clouds: 

1. Mesh graph-based representation
  - separator factorization GFI `SeparationGFIntegrator`
  - trees approximating graph metric
    - FRT trees-based `FRTTreeGFIntegrator`
    - Bartal trees-based GFI `BartalTreeGFIntegrator`
    - spanning tree-based GFI `SpanningTreeGFIntegrator`
2. $\epsilon$-NN (Nearest Neighbor) based representation
  - random feature diffusion GFI `DFGFIntegrator`

These GFIs can be readily used for the following tasks
- interpolation task using `Interpolator`
  - by specifying parameters`GFIntegrator, vertices_known, vertices_interpolate` at instantiation
  - and after, calling method `interpolate` while specifying the field values on the `vetrices_known`
- Wasserstein barycenter using `ConvolutionalBarycenter`
  - by specifying parameters `niter, tolerance` at instantiation
  - and after, calling method `get_convolutional_barycenter` while specifying array with distributions, mixing weights and `GFIntegrator.integrate_graph_field` 

## Experiments

### Vertex normal prediction

First download Thingi10K mesh data from this [link](https://ten-thousand-models.appspot.com/). The mesh IDs we used in our paper are listed in Appendix C1 of our paper.

```scripts/experiments/vertex_normal_prediction_config.yaml``` is an example configuration file to run vertex normal prediction task.

To run experiment on this task:
```sh
python scripts/experiments/vertex_normal_prediction.py 
```

For information on how to run each experiment:

* [scripts/experiments/gaussian_kernel_test.py](docs/experiments/gaussian_kernel_test.md)
* [scripts/experiments/meshgraphdata_interpolator_3dplot.py](docs/experiments/meshgraphdata_interpolator_3dplot.md)
* [scripts/experiments/placebo_separator_test.py](docs/experiments/placebo_separator_test.md)
* [scripts/experiments/separator_test.py](docs/experiments/separator_test.md)
* [scripts/experiments/graph_diffusion_gf_integrator_test.py](docs/experiments/graph_diffusion_integrator_test.md)


### Wasserstein barycenter

Follow the instructions above to download Thingi10K mesh [data](https://ten-thousand-models.appspot.com/).

To run experiments for RFD:
```sh
python scripts/experiments/compare_dfgf_bfgf_wass_barycenter.py
```

To run experiments for SF:
```sh
python scripts/experiments/compare_psgf_bfgf_wass_barycenter.py 
```

To run experiments for trees:
```sh
python scripts/experiments/bf_tspan_sf_wass_barycenter.py
```

### (Fused) Gromov-Wasserstein discrepancy 

To run experiments for RFD on GW with conjugate gradient method:
```sh
python scripts/experiments/test_gromov_wasserstein.py
```

To run experiments for RFD on GW with proximal method:
```sh
python scripts/experiments/test_gromov_wasserstein_discrepancy.py
```

To run experiments for RFD on FGW:
```sh
python scripts/experiments/test_fgw_diffusion.py
```

To run experiments for SF on GW with conjugate gradient method:
```sh
python scripts/experiments/test_gw_separator.py
```

To run experiments for SF on GW with proximal method:
```sh
python scripts/experiments/test_gw_discrepancy_separator.py
```

To run experiments for RFD on FGW:
```sh
python scripts/experiments/test_fgw_separator.py
```

## MeshGraphNet datasets
For information on how to download and prepare meshgraphnet dataset:

* [docs/experiments/prepare_graph_mesh_data.md](docs/experiments/prepare_graph_mesh_data.md)
