# Prepare MeshGraphNet Dataset 

This document contains steps to generate flag-simple graph dataset used in paper [LEARNING MESH-BASED SIMULATION
WITH GRAPH NETWORKS](https://arxiv.org/pdf/2010.03409.pdf). 

Since the original paper is implemented in Tensorflow 1, we found this repo [PyTorch version of Learning Mesh-Based Simulation with Graph Networks (ICLR 2021)](https://github.com/wwMark/meshgraphnets), which contains Pytorch implementation to be pretty useful. 

The specific steps to generate the dataset are as follows:

Download dataset: 

    mkdir -p ${DATA}
    bash scripts/prepare_meshgraphnet_datasets/download_meshgraphnet_dataset.sh flag_simple ${DATA}

Please set the path ${DATA} the same as ega.default_meshgraphnet_dataset_path.

Go to the dataset directory and generate .idx file(needed by package tfrecord for reading .tfrecord file in PyTorch):

    python -m tfrecord.tools.tfrecord2idx <file>.tfrecord <file>.id

where \<file\> is one of train, valid and test.

After finishing these two steps, we can run the folowing script to generate mesh graph data for flag-simple:

```sh
python scripts/prepare_meshgraphnet_datasets/generate_meshgraphnet_data.py 
```

`--model`: currently support cloth only 

`--rollout_split`: the original dataset is splitted into train, valid, and test. You can choose one of them to generate the mesh graph data.

`--dataset`: currently support flag_simple only 

`--trajectories`: number of trajectories to generate. The default is 5. 

`--snapshot_frequency`: each trajectory contains data from time t=0 until t=400. You can set different values of this parameter to generate data with different sample frequency.


