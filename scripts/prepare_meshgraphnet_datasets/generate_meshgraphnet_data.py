# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================]

"""
This file generates mesh graphs from the dataset used in paper: https://arxiv.org/pdf/2010.03409.pdf
Most of the code here are refactorized from the pytorch implementation of this paper here: https://github.com/wwMark/meshgraphnets
"""

import argparse 
import os 
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import IterableDataset, DataLoader
import torch 
import enum
import json 
import numpy as np 
import torch.nn.functional as F
import collections 
import trimesh
import pickle
from typing import List, Callable
# mesh graph data should be put under this path 
from ega import default_meshgraphnet_dataset_path
from ega.util.mesh_utils import neighbors_in_cyclic_order, random_circular_rotation


def get_args_parser():
    parser = argparse.ArgumentParser('GraphMeshNet', add_help=False)
    
    parser.add_argument('--model', default='cloth', type=str,
        choices = ['cloth', 'deform'], help='Select model to run.')
    parser.add_argument('--rollout_split', default='valid', type=str,
        choices =  ['train', 'test', 'valid'], help='Which dataset split to use for rollouts.')
    parser.add_argument('--dataset', default='flag_simple', type=str,
        choices =  ['flag_simple', 'deforming_plate'])    
    parser.add_argument('--trajectories', type = int, default = 5, help = 'No. of training trajectories')
    parser.add_argument('--snapshot_frequency', type = int, default = 20, help = 'frequency to get snapshots for meshgraph data per trajectory')
    parser.add_argument('--device', default='cpu', type=str,
        choices =  ['cpu', 'cuda'], help='use default cuda or cpu')
    
    return parser 


class NodeType(enum.IntEnum):
    """
    This class specifies different node types used in the dataset.
    referenced from common.py in the Pytorch repo https://github.com/wwMark/meshgraphnets
    """
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


""" referenced from run_model.py from the Pytorch repo """
PARAMETERS = {
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, loss_type='cloth',
                  stochastic_message_passing_used='False')
}


class FlagSimpleDatasetIterative(IterableDataset):
    """
    Iterable dataset to reduce main memory usage, no multiprocessing
    """
    def __init__(self, path, split, add_targets=False, split_and_preprocess=False):
        self.path = path
        self.split = split
        self._add_targets = add_targets
        self._split_and_preprocess = split_and_preprocess
        '''
        self.add_targets = add_targets
        self.split_and_preprocess = split_and_preprocess
        '''

        tfrecord_path = os.path.join(path, split + ".tfrecord")
        # index is generated by tfrecord2idx
        index_path = os.path.join(path, split + ".idx")
        tf_dataset = TFRecordDataset(tfrecord_path, index_path, None)
        # loader and iter(loader) have size 1000, which is the number of all training trajectories
        loader = torch.utils.data.DataLoader(tf_dataset, batch_size=1)
        # use list to make list from iterable so that the order of elements is ensured
        self.dataset = iter(loader)

    def __iter__(self):
        return self.dataset

    def add_targets(self):
        """Adds target and optionally history fields to dataframe."""
        fields = 'world_pos'
        add_history = True

        def fn(trajectory):
            out = {}
            for key, val in trajectory.items():
                out[key] = val[1:-1]
                if key in fields:
                    if add_history:
                        out['prev|' + key] = val[0:-2]
                    out['target|' + key] = val[2:]
            return out

        return fn

    def split_and_preprocess(self):
        """Splits trajectories into frames, and adds training noise."""
        noise_field = 'world_pos'
        noise_scale = 0.003
        noise_gamma = 0.1
        device = args.device 

        def add_noise(frame):
            zero_size = torch.zeros(frame[noise_field].size(), dtype=torch.float32).to(device)
            noise = torch.normal(zero_size, std=noise_scale).to(device)
            other = torch.Tensor([NodeType.NORMAL.value]).to(device)
            mask = torch.eq(frame['node_type'], other.int())[:, 0]
            mask = torch.stack((mask, mask, mask), dim=1)
            noise = torch.where(mask, noise, torch.zeros_like(noise))
            frame[noise_field] += noise
            frame['target|' + noise_field] += (1.0 - noise_gamma) * noise
            return frame

        def element_operation(trajectory):
            world_pos = trajectory['world_pos']
            mesh_pos = trajectory['mesh_pos']
            node_type = trajectory['node_type']
            cells = trajectory['cells']
            target_world_pos = trajectory['target|world_pos']
            prev_world_pos = trajectory['prev|world_pos']
            trajectory_steps = []
            for i in range(399):
                wp = world_pos[i]
                mp = mesh_pos[i]
                twp = target_world_pos[i]
                nt = node_type[i]
                c = cells[i]
                pwp = prev_world_pos[i]
                trajectory_step = {'world_pos': wp, 'mesh_pos': mp, 'node_type': nt, 'cells': c,
                                   'target|world_pos': twp, 'prev|world_pos': pwp}
                noisy_trajectory_step = add_noise(trajectory_step)
                trajectory_steps.append(noisy_trajectory_step)
            return trajectory_steps

        return element_operation


def load_dataset(path, split, add_targets=False, split_and_preprocess=False, batch_size=1, prefetch_factor=2):
    """ this function returns a torch dataloader """
    return DataLoader(FlagSimpleDatasetIterative(path=path, split=split, add_targets=add_targets, 
          split_and_preprocess=split_and_preprocess), batch_size=batch_size, prefetch_factor=prefetch_factor, 
          shuffle=False, num_workers=0)


def triangles_to_edges(faces, deform=False):
    """
    Computes mesh edges from triangles.
    referenced from common.py of the Pytorch repo https://github.com/wwMark/meshgraphnets
    """
    if not deform:
        # collect edges from triangles
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           torch.stack((faces[:, 2], faces[:, 0]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
        return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}
    else:
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           faces[:, 2:4],
                           torch.stack((faces[:, 3], faces[:, 0]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
        return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}
    

def add_targets(params):
    """Adds target and optionally history fields to dataframe."""
    fields = params['field']
    add_history = params['history']
    loss_type = params['loss_type']

    def fn(trajectory):
        if loss_type == 'deform':
            out = {}
            for key, val in trajectory.items():
                out[key] = val[0:-1]
                if key in fields:
                    out['target|' + key] = val[1:]
                if key == 'stress':
                    out['target|stress'] = val[1:]
            return out
        elif loss_type == 'cloth':
            out = {}
            for key, val in trajectory.items():
                out[key] = val[1:-1]
                if key in fields:
                    if add_history:
                        out['prev|' + key] = val[0:-2]
                    out['target|' + key] = val[2:]
            return out
    return fn


def split_and_preprocess(params, model_type):
    """Splits trajectories into frames, and adds training noise."""
    noise_field = params['field']
    noise_scale = params['noise']
    noise_gamma = params['gamma']

    def add_noise(frame):
        zero_size = torch.zeros(frame[noise_field].size(), dtype=torch.float32).to(args.device)
        noise = torch.normal(zero_size, std=noise_scale).to(args.device)
        other = torch.Tensor([NodeType.NORMAL.value]).to(args.device)
        mask = torch.eq(frame['node_type'], other.int())[:, 0]
        mask_sequence = []
        for i in range(noise.shape[1]):
            mask_sequence.append(mask)
        mask = torch.stack(mask_sequence, dim=1)
        noise = torch.where(mask, noise, torch.zeros_like(noise))
        frame[noise_field] += noise
        frame['target|' + noise_field] += (1.0 - noise_gamma) * noise
        return frame

    def element_operation(trajectory):
        trajectory_steps = []
        for i in range(steps):
            trajectory_step = {}
            for key, value in trajectory.items():
                trajectory_step[key] = value[i]
            noisy_trajectory_step = add_noise(trajectory_step)
            trajectory_steps.append(noisy_trajectory_step)
        return trajectory_steps

    return element_operation


def squeeze_data_frame(data_frame):
    for k, v in data_frame.items():
        data_frame[k] = torch.squeeze(v, 0)
    return data_frame


def process_trajectory(trajectory_data, params, model_type, dataset_dir, add_targets_bool=False,
                       split_and_preprocess_bool=False):
    global steps

    try:
        with open(os.path.join(dataset_dir, 'meta.json'), 'r') as fp:
            meta = json.loads(fp.read())
        shapes = {}
        dtypes = {}
        types = {}
        steps = meta['trajectory_length'] - 2
        for key, field in meta['features'].items():
            shapes[key] = field['shape']
            dtypes[key] = field['dtype']
            types[key] = field['type']
    except FileNotFoundError as e:
        print(e)
        quit()
    trajectory = {}
    # decode bytes into corresponding dtypes
    for key, value in trajectory_data.items():
        raw_data = value.numpy().tobytes()
        mature_data = np.frombuffer(raw_data, dtype=getattr(np, dtypes[key]))
        mature_data = torch.from_numpy(mature_data).to(args.device)
        reshaped_data = torch.reshape(mature_data, shapes[key])
        if types[key] == 'static':
            reshaped_data = torch.tile(reshaped_data, (meta['trajectory_length'], 1, 1))
        elif types[key] == 'dynamic_varlen':
            pass
        elif types[key] != 'dynamic':
            raise ValueError('invalid data format')
        trajectory[key] = reshaped_data

    if add_targets_bool:
        trajectory = add_targets(params)(trajectory)
    if split_and_preprocess_bool:
        trajectory = split_and_preprocess(params, model_type)(trajectory)
    return trajectory




def faces_to_adjacency_matrices(faces: np.ndarray, nb_nodes: int, seed = 0) -> List[List[int]]:
    """
    this function is heavily built on the original trimesh_to_adjacency_matrices, 
    the only difference is that the input here replaces trimesh into faces, which is a 2d array of size N by 3.
    Each row of the faces matrix represents the 3 vertices of that face 
    """
    vertices = list(range(nb_nodes))
    faces_adj_to_vertices = []
    adjacency_lists = []
    for _  in range(len(vertices)):
        faces_adj_to_vertices.append([])  
        adjacency_lists.append([])  
    for index in range(len(faces)):
        v1, v2, v3 = faces[index]
        faces_adj_to_vertices[v1].append(index)
        faces_adj_to_vertices[v2].append(index)
        faces_adj_to_vertices[v3].append(index)
    for index in range(len(faces_adj_to_vertices)):
        #print(index)
        vertex_adjacency_list = []
        edge_dict = dict()
        rev_edge_dict = dict()
        first_vertex = 0
        for face_index in faces_adj_to_vertices[index]:
            x, y = neighbors_in_cyclic_order(faces[face_index], index)
            edge_dict[x] = y
            rev_edge_dict[y] = x
            first_vertex = x
        next_vertex = first_vertex
        while True:
            mesh_edge_reached = False
            vertex_adjacency_list.append(next_vertex)
            if not next_vertex in edge_dict:
                mesh_edge_reached = True
            else:
                next_vertex = edge_dict[next_vertex]
            if next_vertex == first_vertex and not mesh_edge_reached:
                break
            elif mesh_edge_reached:
                vertex_adjacency_list = []
                while True:
                    vertex_adjacency_list.append(next_vertex)
                    if next_vertex in rev_edge_dict:
                        next_vertex = rev_edge_dict[next_vertex] 
                    else:
                        break  
                vertex_adjacency_list.reverse() 
                break 
        adjacency_lists[index] = vertex_adjacency_list
    return random_circular_rotation(adjacency_lists, seed)



def build_graph(inputs):
    """
    Builds input graph.
    Current version will return a weighted graph (distance as weights) with node features (velocity and node types).
    """
    world_pos = inputs['world_pos']
    prev_world_pos = inputs['prev|world_pos']
    node_type = inputs['node_type']
    velocity = world_pos - prev_world_pos
    one_hot_node_type = F.one_hot(node_type[:, 0].to(torch.int64), NodeType.SIZE)
    node_features = torch.cat((velocity, one_hot_node_type), dim=-1)
    node_features = np.array(node_features) # we use numpy in this version. 
    cells = inputs['cells']
    
    n_nodes = len(node_features)
    vertices = list(range(n_nodes))
    adjacency_list = faces_to_adjacency_matrices(cells.numpy(), n_nodes)
    

    weight_list = [[] for i in range(n_nodes)]
    for i, neighbors_i in enumerate(adjacency_list):
        for nb in neighbors_i:
            dist = np.linalg.norm(world_pos[i] - world_pos[nb])
            weight_list[i].append(dist)
    
    """
    # we use trimesh here to get the triangle edges list of each node. 
    # we can also use the following code for visualization 
    mesh = trimesh.Trimesh(vertices = np.array(world_pos), faces = np.array(cells), process = False)
    mesh.show()
    """
    return {'adjacency_list':adjacency_list, 
            'weight_list':weight_list, 
            'node_features':node_features, 
            'vertices':vertices,
            'world_pos': np.array(world_pos), 
            'prev_world_pos': np.array(prev_world_pos),
            'faces': np.array(cells)}


def main():
    dataset_dir = os.path.join(default_meshgraphnet_dataset_path, args.dataset)
    params = PARAMETERS[args.model]
    
    ds_loader = load_dataset(dataset_dir, split=args.rollout_split, add_targets=True, split_and_preprocess=True)
    ds_iterator = iter(ds_loader)

    processed_path = os.path.join(default_meshgraphnet_dataset_path, args.dataset, 'processed_data')
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
                
    for trajectory_index in range(args.trajectories):
        
        trajectory = next(ds_iterator)
        trajectory = process_trajectory(trajectory, params, args.model, dataset_dir, True, True)
        meshgraph_list = []
        
        for data_frame_index, data_frame in enumerate(trajectory):
            if data_frame_index % args.snapshot_frequency == 0:
                data_frame = squeeze_data_frame(data_frame)
                meshgraph = build_graph(data_frame)
                meshgraph_list.append(meshgraph)
            
        processed_file = os.path.join(processed_path, "trajectory_{}.pkl".format(trajectory_index))
        with open (processed_file, 'wb') as f:
            pickle.dump(meshgraph_list, f)
        print("trajectory {} saved".format(trajectory_index))

    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser('GraphMeshNet', parents=[get_args_parser()])
    args = parser.parse_args()

    main() 
