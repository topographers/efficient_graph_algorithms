#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import os
import copy 
import trimesh 
import time 
import torch
import scipy 
import pathlib
import yaml
from typing import List 
from sklearn.neighbors import kneighbors_graph

from ega.algorithms.expm32 import expm32
from ega.algorithms.expm64 import expm64
from ega.algorithms.brute_force import BFGFIntegrator
from ega.algorithms.separation_gf_integrator import SeparationGFIntegrator 
from ega.algorithms.frt_trees import FRTTreeGFIntegrator
from ega.algorithms.bartal_trees import BartalTreeGFIntegrator
from ega.algorithms.spanning_trees import SpanningTreeGFIntegrator
from ega.algorithms.graph_diffusion_gf_integrator import DFGFIntegrator

from ega.util.lanczos import lanczos
from ega.util.sdd_solver import *
from ega.util.gaussian_kernel import GaussianKernel 
from ega.util.interpolator import Interpolator
from ega.util.mesh_utils import calculate_interpolation_metrics
from ega.util.mesh_utils import random_projection_creator, density_function, fourier_transform, trimesh_to_adjacency_matrices
from ega.util.graphs_networkx_utils import *


def generate_weights_from_adjacency_list(adjacency_lists: List[List[int]], positions: np.ndarray = None, 
                                         unweighted = False) -> List[List[int]]:
    """
    given an adjacency list, this function will return the corresponding unweighted list 
    (every element equal to 1 in this unweighted list)
    """
    weight_lists = []
    for i, list_i in enumerate(adjacency_lists):
        current_list = []
        for j in list_i:
            if positions is None:
                current_list.append(1)
            else:
                dist = np.linalg.norm(positions[i] - positions[j])
                current_list.append(dist)
        if unweighted:
            weight_lists.append(len(current_list) * [1])
        else:
            weight_lists.append(current_list)

    return weight_lists


def reset_mask_ratio(mesh_data, mask_ratio, unweighted = False):

    adjacency_list = trimesh_to_adjacency_matrices(mesh_data)
    weight_list = generate_weights_from_adjacency_list(adjacency_list, mesh_data.vertices, unweighted = unweighted)
    
    vertices =  list(np.arange(mesh_data.vertices.shape[0]))  #mesh_data.vertices # mesh_data['vertices']
    all_fields = mesh_data.vertex_normals # mesh_data['node_features'][:,:3] # the first three columns represent velocities 
    
    world_pos = mesh_data.vertices # mesh_data['world_pos']
    n_vertices = len(vertices)
        
    # divide vertices into known vertices and vertices to be interpolated
    random.seed(0)
    vertices_interpolate = random.sample(vertices, int(mask_ratio * n_vertices))
    vertices_known = list(set(vertices) - set(vertices_interpolate))
    true_fields = all_fields[vertices_interpolate]
    vertices_interpolate_pos = world_pos[vertices_interpolate]
    n_vertices_interpolate = len(vertices_interpolate)
    
    # field_known represents the existed fields on the manifold
    # our goal is to predict the fields that is unknown (the points to be interpolated)
    field_known = copy.deepcopy(all_fields)
    field_known[vertices_interpolate] = 0 
    ones_known = np.ones(shape = (len(field_known), 1)) 
    ones_known[vertices_interpolate] = 0
    
    return adjacency_list, weight_list, vertices, all_fields, world_pos, n_vertices, \
        vertices_interpolate, vertices_known, true_fields, vertices_interpolate_pos, n_vertices_interpolate, \
        field_known, ones_known


def is_psd(mat):
    """
    Check if the matrix is positive semi-definite.
    Note in PSD def, we want the matrices are symmetric otherwise it will lead to a lot of issues.
    For example : see https://math.stackexchange.com/questions/83134/does-non-symmetric-positive-definite-matrix-have-positive-eigenvalues
    """
    return bool((mat == mat.T).all() and (torch.linalg.eigvalsh(mat)[0] >= 0))


def compute_lanczos_matrix_exp(
    A, v, k, use_reorthogonalization=False, return_exp=False
):
    """
    Compute the action of matrix exponential on a vector v using the Lanczos algorithm.
    Can also optionally return the approximate exponential matrix too
    This is figure 4 in https://arxiv.org/abs/1111.1491
    A is assumed to be of shape B x N x N, a batch of symmetric PSD.
    v is assumed to be of shape B x N
    Compute :
        A vector u that is an approximation to exp(-A)v.
    """

    if len(v.shape) == 1:
        v = v.unsqueeze(0)

    # normalize v
    norm_v = torch.linalg.norm(v, dim=1, keepdim=True)
    v = v / norm_v

    # compute Q, T via Lanczos, T is the tridiagonal matrix with shape k x k and Q is of shape n x k
    T, Q = lanczos(
        A, num_eig_vec=k, mask=None, use_reorthogonalization=use_reorthogonalization
    )

    D, P = torch.linalg.eigh(T)
    exp_T = torch.bmm(torch.bmm(P, torch.diag_embed(torch.exp(-D))), P.transpose(1, 2))

    # compute the action
    exp_A = torch.bmm(torch.bmm(Q, exp_T), Q.transpose(1, 2))

    w = torch.einsum("ijk, ik -> ij", exp_A, v) * norm_v

    if return_exp is False:
        return w
    else:
        return w, exp_A


def main(config):
    
    meshgraph_file_ids = config['meshgraph_file_ids']
    log_file = config['log_file']
    method_list = config['method_list']
    meshgraph_path = config['meshgraph_path']
    mask_ratio = config['mask_ratio']
    result_file = config['result_file']
        
    result_list = []
    
    for mesh_id in meshgraph_file_ids:
        
        with open(log_file, 'a') as f:
            f.write("\n\n\n")
            f.write("### mesh_id = " + str(mesh_id))
        
        meshgraph_file = os.path.join(meshgraph_path, str(mesh_id) + '.stl' )
        
        mesh_data = trimesh.load(meshgraph_file, include_normals=True)
        mesh_data.vertices = mesh_data.vertices - mesh_data.vertices.mean(axis = 0) 
        mesh_data.vertices = mesh_data.vertices / np.linalg.norm(mesh_data.vertices, axis = 1).max()
    
        adjacency_list, weight_list, vertices, all_fields, world_pos, n_vertices, \
            vertices_interpolate, vertices_known, true_fields, vertices_interpolate_pos, n_vertices_interpolate, \
            field_known, ones_known = reset_mask_ratio(mesh_data, mask_ratio)
        
        with open(log_file, 'a') as f:
            f.write(", number of nodes = {} ###".format((n_vertices)))
        
        
        ###########################################################################
        ##########################  brute force integrator  #######################
        ###########################################################################
        
        if 'brute_force' in method_list:
            best_cosine_similarity = -1 
            best_sigma = None 
            best_preprosess_time = None 
            best_interpolation_time = None 
            
            frobenius_norm_list = []
            cosine_similarity_list = []
            
            f_fun = GaussianKernel(100)
            
            ## pre-processing time 
            start = time.time() 
            brute_force = BFGFIntegrator(adjacency_list, weight_list, vertices, f_fun, apply_f_fun=False)
            dist_mat = brute_force._m_matrix
            brute_force._f_fun = GaussianKernel(100)
            brute_force._m_matrix = brute_force._f_fun(dist_mat)
            interpolator_bf = Interpolator(brute_force, vertices_known, vertices_interpolate)
            preprocess_time = time.time() - start
            with open(log_file, 'a') as f:
                f.write("\tPreprocessing takes: {} seconds".format(time.time() - start))
            
            bf_sigma_list = config['brute_force']['bf_sigma_list'] 
            
            for sigma in bf_sigma_list: 
                f_fun = GaussianKernel(sigma)
                
                interpolator_bf.integrator._f_fun = f_fun
                interpolator_bf.integrator._m_matrix = f_fun(dist_mat)
                
                ## interpolation time 
                start = time.time() 
                interpolated_fields_bf = interpolator_bf.interpolate(copy.deepcopy(field_known)) / \
                                                interpolator_bf.interpolate(copy.deepcopy(ones_known))
                interpolation_time = time.time() - start
               
                frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_bf)
                with open(log_file, 'a') as f:
                    f.write("\n")
                    f.write("##### brute force GF Integrator  -- sigma = {} #####".format(sigma))
                    f.write("\tInterpolation takes: {} seconds".format(time.time() - start))
                    f.write("\tcosine_similarity is: {}".format(cosine_similarity))
                    f.write("\tfrobenius_norm is: {}".format(frobenius_norm))
                
                if cosine_similarity > best_cosine_similarity:
                    best_cosine_similarity = cosine_similarity 
                    best_sigma = sigma 
                    best_preprocess_time = preprocess_time 
                    best_interpolation_time = interpolation_time
                    
                frobenius_norm_list.append(frobenius_norm)
                cosine_similarity_list.append(cosine_similarity)
        
            with open(log_file, 'a') as f:
                f.write("\n")
                f.write("$$$$ Best sigma is: {}, best cosine similarity is: {}".format(best_sigma, best_cosine_similarity))
            print("@@@@ Best sigma is: {}, best cosine similarity is: {}".format(best_sigma, best_cosine_similarity))
        
            result_list.append([best_cosine_similarity, best_preprocess_time, best_interpolation_time])
            np.savetxt(result_file, result_list, delimiter=",")            
        
        
        ###########################################################################
        #######################  brute force on knn graph  ########################
        ###########################################################################
    
        if 'brute_force_knn' in method_list:
            best_cosine_similarity = -1 
            best_sigma = None 
            best_knn = None  
            best_preprosess_time = None 
            best_interpolation_time = None 
            
            bfknn_knn_list = config['brute_force_knn']['bfknn_knn_list']
            bfknn_sigma_list = config['brute_force_knn']['bfknn_sigma_list']
            
            for k_nn in bfknn_knn_list:
                
                A = kneighbors_graph(world_pos, k_nn, mode='connectivity', include_self=True).toarray()
                AAT = ((A + A.T)==2).astype(int) 
                
                adjacency_list_knn = []
                weight_list_knn = []
                
                for i in range(len(AAT)):
                    neighbors = np.where(AAT[i]==1)[0].tolist()
                    adjacency_list_knn.append(neighbors)
                weight_list_knn = generate_weights_from_adjacency_list(adjacency_list_knn, mesh_data.vertices)    
                
                f_fun = GaussianKernel(100)
                
                ## pre-processing time 
                start = time.time() 
                brute_force = BFGFIntegrator(adjacency_list_knn, weight_list_knn, vertices, f_fun, apply_f_fun=False)
                dist_mat = brute_force._m_matrix
                brute_force._f_fun = GaussianKernel(100)
                brute_force._m_matrix = brute_force._f_fun(dist_mat)
                interpolator_bf = Interpolator(brute_force, vertices_known, vertices_interpolate)
                preprocess_time = time.time() - start
                with open(log_file, 'a') as f:
                    f.write("\tPreprocess: {}".format(preprocess_time))
                
                for sigma in bfknn_sigma_list:
                    f_fun = GaussianKernel(sigma)
                    
                    interpolator_bf.integrator._f_fun = f_fun
                    interpolator_bf.integrator._m_matrix = f_fun(dist_mat)
                    
                    ## interpolation time 
                    start = time.time() 
                    interpolated_fields_bf = interpolator_bf.interpolate(copy.deepcopy(field_known)) / \
                                                    (1e-9 + interpolator_bf.interpolate(copy.deepcopy(ones_known)))
                    interpolation_time = time.time() - start
                   
                    ## statistics
                    frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_bf)
                    with open(log_file, 'a') as f:
                        f.write("\n")
                        f.write("## BF (knn) -- sigma = {}, k_nn = {} ##".format(sigma, k_nn))
                        f.write("\tInterpolation: {}".format(interpolation_time))
                        f.write("\tcosine: {}".format(np.round(cosine_similarity, 4)))
                        f.write("\tfro: {}".format(np.round(frobenius_norm, 4)))
                    
                    if cosine_similarity > best_cosine_similarity:
                        best_cosine_similarity = cosine_similarity 
                        best_sigma = sigma 
                        best_knn = k_nn
                        best_preprocess_time = preprocess_time 
                        best_interpolation_time = interpolation_time
                        
            with open(log_file, 'a') as f:
                f.write("\n")
                f.write("$$$$ Best sigma: {}, best k_nn: {}, best cosine: {}, process time: {}, interpolate time: {}".format(best_sigma, best_knn,  np.round(best_cosine_similarity, 4), best_preprocess_time, best_interpolation_time))
           
            result_list.append([best_cosine_similarity, best_preprocess_time, best_interpolation_time])
            np.savetxt(result_file, result_list, delimiter=",")
    
        
        ###########################################################################
        #####################  brute force on epsilon graph  ######################
        ###########################################################################
        
        if 'brute_force_eps' in method_list:
            best_cosine_similarity = -1 
            best_sigma = None 
            best_eps = None  
            best_preprosess_time = None 
            best_interpolation_time = None 
            
            bfeps_epsilon_list = config['brute_force_eps']['bfeps_epsilon_list']
            bfeps_sigma_list = config['brute_force_eps']['bfeps_sigma_list']
            
            for epsilon in bfeps_epsilon_list:
            
                AAT = (scipy.spatial.distance.cdist(world_pos, world_pos) < epsilon).astype(int)
                
                adjacency_list_knn = []
                weight_list_knn = []
                
                for i in range(len(AAT)):
                    neighbors = np.where(AAT[i]==1)[0].tolist()
                    adjacency_list_knn.append(neighbors)

                weight_list_knn = generate_weights_from_adjacency_list(adjacency_list_knn, mesh_data.vertices)  
                f_fun = GaussianKernel(100)
    
                ## pre-processing time 
                start = time.time() 
                brute_force = BFGFIntegrator(adjacency_list_knn, weight_list_knn, vertices, f_fun, apply_f_fun=False)
                dist_mat = brute_force._m_matrix
                brute_force._f_fun = GaussianKernel(100)
                brute_force._m_matrix = brute_force._f_fun(dist_mat)
                interpolator_bf = Interpolator(brute_force, vertices_known, vertices_interpolate)
                preprocess_time = time.time() - start
                with open(log_file, 'a') as f:
                    f.write("\tPreprocess: {}".format(preprocess_time))
                
                for sigma in bfeps_sigma_list:
                    
                    f_fun = GaussianKernel(sigma)
                    
                    interpolator_bf.integrator._f_fun = f_fun
                    interpolator_bf.integrator._m_matrix = f_fun(dist_mat)
                    
                    ## interpolation time 
                    start = time.time() 
                    interpolated_fields_bf = interpolator_bf.interpolate(copy.deepcopy(field_known)) / \
                                                    (1e-9 + interpolator_bf.interpolate(copy.deepcopy(ones_known)))
                    interpolation_time = time.time() - start
                    
                    ## statistics
                    frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_bf)
                    with open(log_file, 'a') as f:
                        f.write("\n")
                        f.write("## BF (eps) -- sigma = {}, eps = {} ##".format(sigma, epsilon))
                        f.write("\tInterpolation: {}".format(interpolation_time))
                        f.write("\tcosine: {}".format(np.round(cosine_similarity, 4)))
                        f.write("\tfro: {}".format(np.round(frobenius_norm, 4)))
                    
                    if cosine_similarity > best_cosine_similarity:
                        best_cosine_similarity = cosine_similarity 
                        best_sigma = sigma 
                        best_eps = epsilon
                        best_preprocess_time = preprocess_time 
                        best_interpolation_time = interpolation_time
                        
            with open(log_file, 'a') as f:
                f.write("\n")
                f.write("$$$$ Best sigma: {}, best eps: {}, best cosine: {}, process time: {}, interpolate time: {}".format(best_sigma, best_eps,  np.round(best_cosine_similarity, 4), best_preprocess_time, best_interpolation_time))
        
            result_list.append([best_cosine_similarity, best_preprocess_time, best_interpolation_time])
            np.savetxt(result_file, result_list, delimiter=",")
    
        
        ###########################################################################
        #################  graph-diffusion graph-field integrator  ################
        ###########################################################################
        
        if 'graph_diffusion' in method_list:
            best_cosine_similarity = -1 
            best_eps = None    
            best_lambda_par = None 
            best_num_random_features = None 
            best_preprocess_time = None 
            best_interpolation_time = None 
            
            dim = 3
        
            gdgf_lambda_par_list = config['graph_diffusion']['gdgf_lambda_par_list']
            gdgf_epsilon_list = config['graph_diffusion']['gdgf_epsilon_list']
            gdgf_num_random_features = config['graph_diffusion']['gdgf_num_random_features']
            
            for lambda_par in gdgf_lambda_par_list:
                for epsilon in gdgf_epsilon_list:
                    for num_rand_features in gdgf_num_random_features:
                      
                        ## pre-processing time
                        start = time.time()      
                        dfgf_integrator = DFGFIntegrator(world_pos,  epsilon, lambda_par, 
                                                         num_rand_features, dim, 
                                                         random_projection_creator, density_function, 
                                                         fourier_transform)
                        interpolator_dfgf = Interpolator(dfgf_integrator, vertices_known, vertices_interpolate)
                        preprocess_time = time.time() - start 
                  
                        ## interpolation time
                        start = time.time()        
                        interpolated_fields_dfgf = interpolator_dfgf.interpolate(copy.deepcopy(field_known)) / \
                                                  interpolator_dfgf.interpolate(copy.deepcopy(ones_known))
                        interpolation_time = time.time() - start
                        
                        ## statistics 
                        frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_dfgf)
                        with open(log_file, 'a') as f:
                            f.write("\n")
                            f.write("##### graph diffusion GF Integrator  -- lambda_par = {}, epsilon = {}, num_rand_features = {} #####".format(lambda_par, epsilon, num_rand_features))                
                            f.write("\tPreprocessing takes: {} seconds".format(preprocess_time))
                            f.write("\tInterpolation takes: {} seconds".format(interpolation_time))
                            f.write("\tcosine_similarity is: {}".format(cosine_similarity))
                            f.write("\tfrobenius_norm is: {}".format(frobenius_norm))
                            
                        if cosine_similarity > best_cosine_similarity:
                            best_cosine_similarity = cosine_similarity
                            best_eps = epsilon
                            best_lambda_par = lambda_par
                            best_num_random_features = num_rand_features
                            best_preprocess_time = preprocess_time 
                            best_interpolation_time = interpolation_time 
                            
            with open(log_file, 'a') as f:
                f.write("\n")
                f.write("$$$$ eps: {}, lambda_par: {}, #RF is: {}, cosine: {}, process: {}, interpolate: {}".format(best_eps, best_lambda_par, best_num_random_features, np.round(best_cosine_similarity, 4), best_preprocess_time, best_interpolation_time))
          
            result_list.append([best_cosine_similarity, best_preprocess_time, best_interpolation_time])
            np.savetxt(result_file, result_list, delimiter=",")
    
        
        ###########################################################################
        #################  seperation graph-field integrator  ################
        ###########################################################################
    
        if 'seperator' in method_list:
            best_cosine_similarity = -1 
            best_threshold = None    
            best_unit_size = None 
            best_sigma = None 
            best_preprocess_time = None 
            best_interpolation_time = None 
            
            sgf_threshold_nb_vertices_list = [int(0.25*n_vertices)]
            sgf_unit_size_list = config['seperator']['sgf_unit_size_list']
            sgf_sigma_list = config['seperator']['sgf_sigma_list']
            
            for threshold_nb_vertices in sgf_threshold_nb_vertices_list:
                for unit_size in sgf_unit_size_list:
                    for sigma in sgf_sigma_list:
                        
                        with open(log_file, 'a') as f:
                            f.write("\n")
                            f.write("##### seperation GF Integrator  -- threshold_nb_vertices = {}, unit_size = {}, sigma = {} #####".format(threshold_nb_vertices, unit_size, sigma))                
                        
                        f_fun = GaussianKernel(sigma)
                        
                        try:
                            ## pre-processing time
                            start = time.time()      
                            sgf_integrator = SeparationGFIntegrator(adjacency_list, weight_list, vertices, f_fun, 
                                                                    unit_size=unit_size, threshold_nb_vertices=threshold_nb_vertices)
                            interpolator_sgf = Interpolator(sgf_integrator, vertices_known, vertices_interpolate)
                            processing_time = time.time() - start 
                             
                            ## interpolation time
                            start = time.time()        
                            interpolated_fields_sgf = interpolator_sgf.interpolate(copy.deepcopy(field_known)) / \
                                                      interpolator_sgf.interpolate(copy.deepcopy(ones_known))
                            interpolation_time = time.time() - start 
                           
                            ## statistics 
                            frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_sgf)
                            with open(log_file, 'a') as f:
                                f.write("\tPreprocess: {}".format(processing_time))
                                f.write("\tInterpolation: {}".format(interpolation_time))
                                f.write("\tcosine: {}".format(cosine_similarity))
                                f.write("\tfro: {}".format(frobenius_norm))
                        except:
                            pass
                                            
                        if cosine_similarity > best_cosine_similarity:
                            best_cosine_similarity = cosine_similarity
                            best_threshold = threshold_nb_vertices    
                            best_unit_size = unit_size 
                            best_sigma = sigma 
                            best_preprocess_time = processing_time 
                            best_interpolation_time = interpolation_time 
                                          
            with open(log_file, 'a') as f:
                f.write("\n")
                f.write("$$ threshold: {}, unit_size: {}, sigma: {}, cosine: {}, process: {}, interpolation: {}".format(best_threshold, best_unit_size, best_sigma, best_cosine_similarity, best_preprocess_time, best_interpolation_time))
               
            result_list.append([best_cosine_similarity, best_preprocess_time, best_interpolation_time])
            np.savetxt(result_file, result_list, delimiter=",")
    
        
        ###########################################################################
        ####################  torch.matrix_exp (original graph)  ##################
        ###########################################################################
        
        if 'torch_matrix_exp' in method_list:
            best_cosine_similarity = -1 
            best_sigma = None 
            best_preprocess_time = None 
            best_interpolation_time = None 
            
            frobenius_norm_list = []
            cosine_similarity_list = []
            
            tme_sigma_list = config['torch_matrix_exp']['tme_sigma_list']
            
            for sigma in tme_sigma_list:    
                adjacency_list, weight_list, vertices, all_fields, world_pos, n_vertices, \
                    vertices_interpolate, vertices_known, true_fields, vertices_interpolate_pos, n_vertices_interpolate, \
                    field_known, ones_known = reset_mask_ratio(mesh_data, mask_ratio, unweighted=True)
                
                f_fun = GaussianKernel(sigma)
                
                ## pre-processing time 
                start = time.time() 
                brute_force = BFGFIntegrator(adjacency_list, weight_list, vertices, f_fun, apply_f_fun=False)
                A = (brute_force._m_matrix<=1)*1
                brute_force._m_matrix = torch.matrix_exp( sigma * torch.Tensor(A) )
                interpolator_bf = Interpolator(brute_force, vertices_known, vertices_interpolate)
                processing_time = time.time() - start 
                
                ## interpolation time 
                start = time.time() 
                interpolated_fields_bf = interpolator_bf.interpolate(copy.deepcopy(field_known)) / \
                                                (1e-9 + interpolator_bf.interpolate(copy.deepcopy(ones_known)))
                interpolation_time = time.time() - start 
                
                frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_bf)
                with open(log_file, 'a') as f:
                    f.write("\n")
                    f.write("##### torch.matrix_exp (original graph)  -- sigma = {} #####".format(sigma))
                    f.write("\tPreprocessing takes: {} seconds".format(processing_time))
                    f.write("\tInterpolation takes: {} seconds".format(interpolation_time))
                    f.write("\tcosine_similarity is: {}".format(cosine_similarity))
                    f.write("\tfrobenius_norm is: {}".format(frobenius_norm))
                
                if cosine_similarity > best_cosine_similarity:
                    best_cosine_similarity = cosine_similarity 
                    best_sigma = sigma 
                    best_preprocess_time = processing_time 
                    best_interpolation_time = interpolation_time 
                else:
                    break
                    
                frobenius_norm_list.append(frobenius_norm)
                cosine_similarity_list.append(cosine_similarity)
        
            with open(log_file, 'a') as f:
                f.write("\n")
                f.write("$$ sigma: {}, cosine: {}, process: {}, interpolation: {}".format(best_sigma, best_cosine_similarity, best_preprocess_time, best_interpolation_time))
                
            result_list.append([best_cosine_similarity, best_preprocess_time, best_interpolation_time])
            np.savetxt(result_file, result_list, delimiter=",")
        
        
        ###########################################################################
        ####################  almohy2009  ##################
        ###########################################################################
        
        if 'almohy2009' in method_list:
            best_cosine_similarity = -1 
            best_sigma = None 
            best_preprocess_time = None 
            best_interpolation_time = None 
            
            frobenius_norm_list = []
            cosine_similarity_list = []
            
            tme_sigma_list = config['almohy2009']['tme_sigma_list']
            for sigma in tme_sigma_list:    
                adjacency_list, weight_list, vertices, all_fields, world_pos, n_vertices, \
                    vertices_interpolate, vertices_known, true_fields, vertices_interpolate_pos, n_vertices_interpolate, \
                    field_known, ones_known = reset_mask_ratio(mesh_data, mask_ratio, unweighted=True)
                
                f_fun = GaussianKernel(sigma)
                
                ## pre-processing time 
                start = time.time() 
                brute_force = BFGFIntegrator(adjacency_list, weight_list, vertices, f_fun, apply_f_fun=False)
                A = (brute_force._m_matrix<=1)*1
                brute_force._m_matrix = scipy.linalg.expm( sigma * torch.Tensor(A) )
                interpolator_bf = Interpolator(brute_force, vertices_known, vertices_interpolate)
                processing_time = time.time() - start 
               
                ## interpolation time 
                start = time.time() 
                interpolated_fields_bf = interpolator_bf.interpolate(copy.deepcopy(field_known)) / \
                                                (1e-9 + interpolator_bf.interpolate(copy.deepcopy(ones_known)))
                interpolation_time = time.time() - start 
                
                frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_bf)
                with open(log_file, 'a') as f:
                    f.write("\n")
                    f.write("##### almohy 2009  -- sigma = {} #####".format(sigma))
                    f.write("\tPreprocessing takes: {} seconds".format(processing_time))
                    f.write("\tInterpolation takes: {} seconds".format(interpolation_time))
                    f.write("\tcosine_similarity is: {}".format(cosine_similarity))
                    f.write("\tfrobenius_norm is: {}".format(frobenius_norm))
                
                if cosine_similarity > best_cosine_similarity:
                    best_cosine_similarity = cosine_similarity 
                    best_sigma = sigma 
                    best_preprocess_time = processing_time 
                    best_interpolation_time = interpolation_time 
                else:
                    break
                    
                frobenius_norm_list.append(frobenius_norm)
                cosine_similarity_list.append(cosine_similarity)
        
            with open(log_file, 'a') as f:
                f.write("\n")
                f.write("$$ sigma: {}, cosine: {}, process: {}, interpolation: {}".format(best_sigma, best_cosine_similarity, best_preprocess_time, best_interpolation_time))
                
            result_list.append([best_cosine_similarity, best_preprocess_time, best_interpolation_time])
            np.savetxt(result_file, result_list, delimiter=",")
        
        
        ###########################################################################
        #########################  almohy32 (original graph)  #######################
        ###########################################################################
        
        if 'almohy32' in method_list:
            best_cosine_similarity = -1 
            best_sigma = None 
            best_preprocess_time = None 
            best_interpolation_time = None 
            
            frobenius_norm_list = []
            cosine_similarity_list = []
            
            almohy_sigma_list = config['almohy32']['almohy_sigma_list']
            
            for sigma in almohy_sigma_list:    
                adjacency_list, weight_list, vertices, all_fields, world_pos, n_vertices, \
                    vertices_interpolate, vertices_known, true_fields, vertices_interpolate_pos, n_vertices_interpolate, \
                    field_known, ones_known = reset_mask_ratio(mesh_data, mask_ratio, unweighted=True)
                
                f_fun = GaussianKernel(sigma)
                
                ## pre-processing time 
                start = time.time() 
                brute_force = BFGFIntegrator(adjacency_list, weight_list, vertices, f_fun, apply_f_fun=False)
                A = torch.Tensor((brute_force._m_matrix<=1)*1)
                expmA = expm32(sigma * A).numpy()
                processing_time = time.time() - start
                
                ## interpolation time 
                start = time.time() 
                expmAV = np.einsum('ij,j...->i...', expmA, copy.deepcopy(field_known))[vertices_interpolate]
                expmA1 = np.einsum('ij,j...->i...', expmA, copy.deepcopy(ones_known))[vertices_interpolate]
            
                interpolated_fields_bf = expmAV / expmA1
                interpolation_time = time.time() - start
                
                frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_bf)
                with open(log_file, 'a') as f:
                    f.write("\n")
                    f.write("##### al-mohy  -- sigma = {} #####".format(sigma))
                    f.write("\tPreprocessing takes: {} seconds".format(processing_time))
                    f.write("\tInterpolation takes: {} seconds".format(interpolation_time))
                    f.write("\tcosine_similarity is: {}".format(cosine_similarity))
                    f.write("\tfrobenius_norm is: {}".format(frobenius_norm))
               
                if cosine_similarity > best_cosine_similarity:
                    best_cosine_similarity = cosine_similarity 
                    best_sigma = sigma 
                    best_preprocess_time = processing_time 
                    best_interpolation_time = interpolation_time 
                else:
                    break
                
                frobenius_norm_list.append(frobenius_norm)
                cosine_similarity_list.append(cosine_similarity)
        
            with open(log_file, 'a') as f:
                f.write("\n")
                f.write("$$$$ Best sigma: {}, best cosine: {}, process time: {}, interpolate time: {}".format(best_sigma,  np.round(best_cosine_similarity, 4), best_preprocess_time, best_interpolation_time))
            
            result_list.append([best_cosine_similarity, best_preprocess_time, best_interpolation_time])
            np.savetxt(result_file, result_list, delimiter=",")        
        
        
        ###########################################################################
        #########################  almohy64 (original graph)  #####################
        ###########################################################################
        
        if 'almohy64' in method_list:
            best_cosine_similarity = -1 
            best_sigma = None 
            
            frobenius_norm_list = []
            cosine_similarity_list = []
            
            almohy_sigma_list = config['almohy64']['almohy_sigma_list']
            for sigma in almohy_sigma_list:    
                adjacency_list, weight_list, vertices, all_fields, world_pos, n_vertices, \
                    vertices_interpolate, vertices_known, true_fields, vertices_interpolate_pos, n_vertices_interpolate, \
                    field_known, ones_known = reset_mask_ratio(mesh_data, mask_ratio, unweighted=True)
                
                f_fun = GaussianKernel(sigma)
                
                ## pre-processing time 
                start = time.time() 
                brute_force = BFGFIntegrator(adjacency_list, weight_list, vertices, f_fun, apply_f_fun=False)
                A = torch.Tensor((brute_force._m_matrix<=1)*1)
                expmA = expm64(sigma * A).numpy()

                ## interpolation time 
                start = time.time() 
                expmAV = np.einsum('ij,j...->i...', expmA, copy.deepcopy(field_known))[vertices_interpolate]
                expmA1 = np.einsum('ij,j...->i...', expmA, copy.deepcopy(ones_known))[vertices_interpolate]
            
                interpolated_fields_bf = expmAV / expmA1
                
                frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_bf)
                with open(log_file, 'a') as f:
                    f.write("\n")
                    f.write("##### al-mohy  -- sigma = {} #####".format(sigma))
                    f.write("\tPreprocessing takes: {} seconds".format(time.time() - start))
                    f.write("\tInterpolation takes: {} seconds".format(time.time() - start))
                    f.write("\tcosine_similarity is: {}".format(cosine_similarity))
                    f.write("\tfrobenius_norm is: {}".format(frobenius_norm))
                    
                if cosine_similarity > best_cosine_similarity:
                    best_cosine_similarity = cosine_similarity 
                    best_sigma = sigma 
                    
                frobenius_norm_list.append(frobenius_norm)
                cosine_similarity_list.append(cosine_similarity)
        
            with open(log_file, 'a') as f:
                f.write("\n")
                f.write("$$$$ Best sigma is: {}, best cosine similarity is: {}".format(best_sigma, best_cosine_similarity))
            
            result_list.append([best_cosine_similarity, best_preprocess_time, best_interpolation_time])
            np.savetxt(result_file, result_list, delimiter=",")       
        
        
        ###########################################################################
        #########################  lanczos (symm knn graph)  ######################
        ###########################################################################
                
        num_dim = 16
        if 'lanczos' in method_list:
            best_cosine_similarity = -1 
            best_sigma = None 
            best_knn = None 
            best_num_dim = None 
            best_preprocess_time = None 
            best_interpolation_time = None 
            
            lanczos_knn_list = config['lanczos']['lanczos_knn_list']
            lanczos_sigma_list = config['lanczos']['lanczos_sigma_list']
            lanczos_numdim_list = config['lanczos']['lanczos_numdim_list']
            
            for k_nn in lanczos_knn_list:
                for sigma in lanczos_sigma_list: 
                    ## pre-processing time 
                    start = time.time() 
                    A = kneighbors_graph(world_pos, k_nn, mode='connectivity', include_self=True).toarray()
                    AAT = ((A + A.T)==2).astype(int) 
                    AAT = np.diag(AAT.sum(axis = 0)) - AAT
                    AAT = torch.Tensor(AAT).unsqueeze(0) * sigma * (-1)
                    processing_time = time.time() - start
                    with open(log_file, 'a') as f:
                        f.write("\tPreprocessing takes: {} seconds".format(processing_time))
                   
                    for num_dim in lanczos_numdim_list:
                        
                        ## interpolation time 
                        start = time.time() 
                        field_known = torch.Tensor(field_known)
                        ones_known = torch.Tensor(ones_known)
                        
                        pred_fields = [compute_lanczos_matrix_exp(AAT, field_known[:,i], num_dim) for i in range(3)]
                        pred_fields = torch.stack(pred_fields)[:,0,:].T 
                        pred_fields = pred_fields[vertices_interpolate]
                        
                        pred_ones = compute_lanczos_matrix_exp(AAT, ones_known[:,0], num_dim).T
                        pred_ones = pred_ones[vertices_interpolate]
                        
                        interpolated_fields_lanczos = pred_fields / (1e-9 + pred_ones)
                        interpolated_fields_lanczos = interpolated_fields_lanczos.numpy()#[vertices_interpolate]
                        interpolation_time = time.time() - start
                     
                        ## statistics 
                        frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_lanczos)
                        with open(log_file, 'a') as f:
                            f.write("\n")
                            f.write("##### lanczos Integrator  -- k_nn = {}, sigma = {}, num_dim = {} #####".format(k_nn, sigma, num_dim))                
                            f.write("\tInterpolation takes: {} seconds".format(interpolation_time))
                            f.write("\tcosine_similarity is: {}".format(cosine_similarity))
                            f.write("\tfrobenius_norm is: {}".format(frobenius_norm))
                            
                        if cosine_similarity > best_cosine_similarity:
                            best_cosine_similarity = cosine_similarity 
                            best_sigma = sigma 
                            best_knn = k_nn
                            best_num_dim = num_dim
                            best_preprocess_time = processing_time 
                            best_interpolation_time = interpolation_time 
    
            with open(log_file, 'a') as f:
                f.write("\n")
                f.write("$$$$ sigma: {}, knn: {}, numdim: {}, cosine: {}, process: {}, interpolate: {}".format(best_sigma, best_knn, best_num_dim, best_cosine_similarity, best_preprocess_time, best_interpolation_time))
         
            result_list.append([best_cosine_similarity, best_preprocess_time, best_interpolation_time])
            np.savetxt(result_file, result_list, delimiter=",")        
        
        
        ###########################################################################
        #################  SpanningTree graph-field integrator  ################
        ###########################################################################
        
        if 'SpanningTree' in method_list:
            best_cosine_similarity = -1 
            best_sigma = None 
            best_preprocess_time = None 
            best_interpolation_time = None 
            num_trees = 1
            
            frobenius_norm_list = []
            cosine_similarity_list = []
             
            adjacency_list, weight_list, vertices, all_fields, world_pos, n_vertices, \
                vertices_interpolate, vertices_known, true_fields, vertices_interpolate_pos, n_vertices_interpolate, \
                field_known, ones_known = reset_mask_ratio(mesh_data, mask_ratio)
            
            f_fun = GaussianKernel(1)
            ## pre-processing time 
            start = time.time() 
            spanning_trees = SpanningTreeGFIntegrator(adjacency_list, weight_list, vertices, \
                                                            f_fun, num_trees)
            interpolator_bf = Interpolator(spanning_trees, vertices_known, vertices_interpolate)
            preprocess_time = time.time() - start
            with open(log_file, 'a') as f:
                f.write("\tPreprocessing takes: {} seconds".format(preprocess_time))
            
            spanningtree_sigma_list = config['SpanningTree']['spanningtree_sigma_list']
            for sigma in spanningtree_sigma_list: 
                f_fun = GaussianKernel(sigma)
                interpolator_bf.integrator._f_fun = f_fun ##### set f_fun
                
                ## interpolation time 
                start = time.time() 
                interpolated_fields_bf = interpolator_bf.interpolate(copy.deepcopy(field_known)) / \
                                                interpolator_bf.interpolate(copy.deepcopy(ones_known))
                interpolation_time = time.time() - start
                
                frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_bf)
                with open(log_file, 'a') as f:
                    f.write("\n")
                    f.write("##### Spanning Tree GF Integrator  -- sigma = {} #####".format(sigma))
                    f.write("\tInterpolation takes: {} seconds".format(interpolation_time))
                    f.write("\tcosine_similarity is: {}".format(cosine_similarity))
                    f.write("\tfrobenius_norm is: {}".format(frobenius_norm))
                
                if cosine_similarity > best_cosine_similarity:
                    best_cosine_similarity = cosine_similarity 
                    best_num_trees = num_trees
                    best_sigma = sigma 
                    best_interpolation_time = interpolation_time
                    best_preprocess_time = preprocess_time
                    
                frobenius_norm_list.append(frobenius_norm)
                cosine_similarity_list.append(cosine_similarity)
        
            with open(log_file, 'a') as f:
                f.write("\n")
                f.write("@@@@ Best sigma is: {}, best cosine similarity is: {}, process {}, interpolate {} ".format(best_sigma, best_cosine_similarity, best_preprocess_time, best_interpolation_time))
        
            result_list.append([best_cosine_similarity, best_preprocess_time, best_interpolation_time])
            np.savetxt(result_file, result_list, delimiter=",")
    
        
        ###########################################################################
        #################  BartalTree graph-field integrator  ################
        ###########################################################################
        
        if 'BartalTree' in method_list:
            best_cosine_similarity = -1 
            best_sigma = None 
            best_num_trees = None
            best_preprocess_time = None 
            best_interpolation_time = None 
            num_trees = 3
            
            frobenius_norm_list = []
            cosine_similarity_list = []
            
            bartaltree_num_trees_list = config['BartalTree']['bartaltree_num_trees_list']
            bartaltree_sigma_list = config['BartalTree']['bartaltree_sigma_list']
            
            for num_trees in bartaltree_num_trees_list:
                adjacency_list, weight_list, vertices, all_fields, world_pos, n_vertices, \
                    vertices_interpolate, vertices_known, true_fields, vertices_interpolate_pos, n_vertices_interpolate, \
                    field_known, ones_known = reset_mask_ratio(mesh_data, mask_ratio)
                    
                f_fun = GaussianKernel(1)
                ## pre-processing time 
                start = time.time() 
                bartal_trees = BartalTreeGFIntegrator(adjacency_list, weight_list, vertices, \
                                                                f_fun, num_trees)
                interpolator_bf = Interpolator(bartal_trees, vertices_known, vertices_interpolate)
                preprocess_time = time.time() - start
                with open(log_file, 'a') as f:
                    f.write("\tPreprocessing takes: {} seconds".format(preprocess_time))
                
                for sigma in bartaltree_sigma_list: 
                    f_fun = GaussianKernel(sigma)
                    interpolator_bf.integrator._f_fun = f_fun ##### set f_fun
                    
                    ## interpolation time 
                    start = time.time() 
                    interpolated_fields_bf = interpolator_bf.interpolate(copy.deepcopy(field_known)) / \
                                                    interpolator_bf.interpolate(copy.deepcopy(ones_known))
                    interpolation_time = time.time() - start 
                    
                    frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_bf)
                    with open(log_file, 'a') as f:
                        f.write("\n")
                        f.write("##### BartalTree GF Integrator  -- sigma = {}, num_trees = {} #####".format(sigma, num_trees))
                        f.write("\tInterpolation takes: {} seconds".format(interpolation_time))
                        f.write("\tcosine_similarity is: {}".format(cosine_similarity))
                        f.write("\tfrobenius_norm is: {}".format(frobenius_norm))
                    
                    if cosine_similarity > best_cosine_similarity:
                        best_cosine_similarity = cosine_similarity 
                        best_num_trees = num_trees
                        best_sigma = sigma 
                        best_interpolation_time = interpolation_time
                        best_preprocess_time = preprocess_time
                        
                    frobenius_norm_list.append(frobenius_norm)
                    cosine_similarity_list.append(cosine_similarity)
            
            with open(log_file, 'a') as f:
                f.write("\n")
                f.write("@@@@ Best sigma is: {}, best num_trees is: {}, best cosine similarity is: {}".format(best_sigma, num_trees, best_cosine_similarity))
            
            result_list.append([best_cosine_similarity, best_preprocess_time, best_interpolation_time])
            np.savetxt(result_file, result_list, delimiter=",")
    
        
        ###########################################################################
        #################  FRTTree graph-field integrator  ################
        ###########################################################################
        
        if 'FRTTree' in method_list:
            best_cosine_similarity = -1 
            best_sigma = None 
            best_num_trees = None
            best_preprocess_time = None 
            best_interpolation_time = None 
            num_trees = 3
            
            frobenius_norm_list = []
            cosine_similarity_list = []
            
            frttree_num_trees_list = config['FRTTree']['frttree_num_trees_list']
            frttree_sigma_list = config['FRTTree']['frttree_sigma_list']
            
            for num_trees in frttree_num_trees_list:
                adjacency_list, weight_list, vertices, all_fields, world_pos, n_vertices, \
                    vertices_interpolate, vertices_known, true_fields, vertices_interpolate_pos, n_vertices_interpolate, \
                    field_known, ones_known = reset_mask_ratio(mesh_data, mask_ratio)
                    
                f_fun = GaussianKernel(50)
                ## pre-processing time 
                start = time.time() 
                frt_trees = FRTTreeGFIntegrator(adjacency_list, weight_list, vertices, \
                                                                f_fun, num_trees)
                interpolator_bf = Interpolator(frt_trees, vertices_known, vertices_interpolate)
                preprocess_time = time.time() - start
                with open(log_file, 'a') as f:
                    f.write("\tPreprocessing takes: {} seconds".format(preprocess_time))
                
                for sigma in frttree_sigma_list: 
                    f_fun = GaussianKernel(sigma)
                    interpolator_bf.integrator._f_fun = f_fun ##### set f_fun
                    
                    ## interpolation time 
                    start = time.time() 
                    interpolated_fields_bf = interpolator_bf.interpolate(copy.deepcopy(field_known)) / \
                                                    interpolator_bf.interpolate(copy.deepcopy(ones_known))
                    interpolation_time = time.time() - start
                    
                    frobenius_norm, cosine_similarity = calculate_interpolation_metrics(true_fields, interpolated_fields_bf)
                    with open(log_file, 'a') as f:
                        f.write("\n")
                        f.write("##### FRTTree GF Integrator  -- sigma = {}, num_trees = {} #####".format(sigma, num_trees))
                        f.write("\tInterpolation takes: {} seconds".format(interpolation_time))
                        f.write("\tcosine_similarity is: {}".format(cosine_similarity))
                        f.write("\tfrobenius_norm is: {}".format(frobenius_norm))
                    
                    if cosine_similarity > best_cosine_similarity:
                        best_cosine_similarity = cosine_similarity 
                        best_num_trees = num_trees
                        best_sigma = sigma 
                        best_interpolation_time = interpolation_time
                        best_preprocess_time = preprocess_time
                        
                    frobenius_norm_list.append(frobenius_norm)
                    cosine_similarity_list.append(cosine_similarity)
        
            with open(log_file, 'a') as f:
                f.write("\n")
                f.write("@@@@ Best sigma is: {}, best num_trees is: {}, best cosine similarity is: {}".format(best_sigma, num_trees, best_cosine_similarity))
        
            result_list.append([best_cosine_similarity, best_preprocess_time, best_interpolation_time])
            np.savetxt(result_file, result_list, delimiter=",")
            

if __name__ == '__main__':
    
    # load configurations from yaml file 
    with open(os.path.join(pathlib.Path(__file__).parent.resolve(), "vertex_normal_prediction_config.yaml"), "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print(config)
    
    main(config)
    
