#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the TP fractions or PPV on realistic data
"""

import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from Bio.PDB import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import os
import networkx as nx

pfam_to_pdb = {"PF00072": {"pdb": {"id": "3ilh",
                                   "chain_id": "A",
                                   "idxs": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123]},
                           "pfam": {"seq": "VLLIDDDDIVNFLNTTIIRTHRVEEIQSVTSGNAAINKLNELYPSIICIDINMPGINGWELIDLFKQHFNKSIVCLLSSSLDPRDQAKAEASDVDYYVSKPLTANALN----",
                                    "idxs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107]}},
               "PF00512_full_no_gapped": {"pdb": {"id": "3dge",
                                                  "chain_id": "A",
                                                  "idxs": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]},
                                          "pfam" : {"seq": "MKTEFIANISHERTPLTAIKAYAETIYNSELDLSTLKEFLEVIIDQSNHLENLLNELLDFSRLE--",
                                                    "idxs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}},
               "PF00595_full_no_gapped": {"pdb": {"id": "1be9",
                                                  "chain_id": "A",
                                                  "idxs": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88]},
                                          "pfam": {"seq": "-IVIHR-GSTGLGFNIVGGEDGE---GIFISFILAGGPADLSGLRKGDQILSVNGVDLRNASHEQAAIALKNAGQTVTII--",
                                                   "idxs": [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]}},
               "PF02518": {"pdb": {"id": "3g7e",
                                   "chain_id": "A",
                                   "idxs": [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159]},
                           "pfam": {"seq": "-DGTGLHHMVFEVVDNAIDAGHCKEIIVTIH---ADNSVSVQDDGRGIPTGIHPHAGGKFDD-NSYKVSGGLHGVGVSVVNALSQKLELVIQRGETEKTGTMVRFWPSLE-",
                                    "idxs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}},
              
              }

def calc_Calpha_dist_matrix(chain, idx_subset):
    """Returns a matrix of C-alpha distances in a (subset of a) chain"""
    idx_subset = set(idx_subset)
    residue_coords = [residue["CA"].coord for i, residue in enumerate(chain.get_residues()) if i in idx_subset]

    return squareform(pdist(residue_coords))


def calc_min_dist_matrix(chain, idx_subset):
    """Returns a matrix of minimum distances between residues in a (subset of a) chain"""
    idx_subset = set(idx_subset)

    return squareform(
        np.array([min([atom_in_res_i - atom_in_res_j for atom_in_res_i in res_i for atom_in_res_j in res_j])
                 for i, res_i in enumerate(chain.get_residues()) if i in idx_subset
                 for j, res_j in enumerate(chain.get_residues()) if j > i and j in idx_subset])
        )


def top_n_contact_mask(scores_matrix, sequence_proximity_mask, n, return_flattened_scores=False):
    sq = squareform(scores_matrix, checks=False)
    sq_proximity = np.logical_not(squareform(sequence_proximity_mask, checks=False))
    sq[sq_proximity] = -np.inf
    if return_flattened_scores:
        return sq
    contact_mask = np.zeros(len(sq), dtype=bool)
    argsrt = np.argsort(sq)[::-1]
    contact_mask[argsrt[:n]] = True
    
    return squareform(contact_mask)


def contact_matrix_comparison(dist_matrix,
                              scores_matrix,
                              max_eucl_dist=8,
                              n_pred=None,
                              min_sequence_dist=5,
                              contact_mask=None,
                              return_data_for_roc=False):
    assert dist_matrix.shape == scores_matrix.shape
    n_residues = dist_matrix.shape[0]
    sequence_proximity_mask = np.abs(
        np.arange(n_residues) - np.arange(n_residues)[:, None]
        ) >= min_sequence_dist
    
    if contact_mask is None:
        eucl_dist_mask = dist_matrix < max_eucl_dist
        contact_mask = np.triu(np.logical_and(eucl_dist_mask, sequence_proximity_mask))        

    if n_pred is None:
        n_contacts = int(np.sum(contact_mask))
        n_pred = n_contacts

    contact_mask_pred = top_n_contact_mask(scores_matrix,
                                           sequence_proximity_mask,
                                           n=n_pred,
                                           return_flattened_scores=return_data_for_roc)
    if return_data_for_roc:
        contact_scores = contact_mask_pred
        return contact_scores, squareform(contact_mask, checks=False)
    
    contact_mask_pred = np.tril(contact_mask_pred)

    contact_mask_true_pos = np.logical_and(contact_mask_pred.T, contact_mask).T
    contact_mask_false_pos = np.logical_xor(contact_mask_true_pos, contact_mask_pred)
    contact_mask_false_neg = np.logical_and(np.logical_not(contact_mask_pred).T, contact_mask)

    full_matrix = contact_mask_true_pos.T * 1. + contact_mask_false_neg * 3 + contact_mask_true_pos * 1. + contact_mask_false_pos * 4.
    full_matrix[full_matrix == 0] = np.nan
    
    return full_matrix


    
def loadscore(path,msa_name):
    mat = np.load(path)
    pfam_idxs = pfam_to_pdb[msa_name]["pfam"]["idxs"]
    mat = mat[np.asarray(pfam_idxs)][:, np.asarray(pfam_idxs)]
    return mat

   

def ComputeDisMat(msa_name,path_tociffolder):
    pdb_id = pfam_to_pdb[msa_name]["pdb"]["id"]
    chain_id = pfam_to_pdb[msa_name]["pdb"]["chain_id"]
    pdb_idxs = pfam_to_pdb[msa_name]["pdb"]["idxs"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PDBConstructionWarning)
        structure = pdb_parser = MMCIFParser().get_structure(pdb_id, path_tociffolder + f"/{pdb_id}.cif")
    
    chain = structure[0][chain_id]
    dist_mat =dist_matrix_func(chain, pdb_idxs)
    return dist_mat


def ComputeContactMaskDistMat(msa_name,path_tociffolder,path_bm_cm_apc):
    pfam_length = len(pfam_to_pdb[msa_name]["pfam"]["seq"])
    dist_mat = ComputeDisMat(msa_name,path_tociffolder)
    pfam_idxs = pfam_to_pdb[msa_name]["pfam"]["idxs"]

    bmDCA_scores_mat = np.load(path_bm_cm_apc)[np.asarray(pfam_idxs)][:, np.asarray(pfam_idxs)]
   
    
    
    cm = contact_matrix_comparison(dist_mat,bmDCA_scores_mat,**contact_matrix_kwargs,n_pred=2 * pfam_length) # Predict 2L contacts or None 
    contactmask = np.tril(np.isfinite(cm)).T
    return contactmask, dist_mat

def ComputePPV(path_score,msa_name,path_tociffolder,path_bm_cm_apc,bl_natseq):

    scores = loadscore(path_score, msa_name)

    contact_mask,dists= ComputeContactMaskDistMat(msa_name,path_tociffolder,path_bm_cm_apc)
    
    pfam_length = len(pfam_to_pdb[msa_name]["pfam"]["seq"])
    n_pred = 2 * pfam_length
    #n_pred = None
    
    if bl_natseq:
        contact_mask = None
    full_matrix = contact_matrix_comparison(dists,
                                scores,
                                contact_mask=contact_mask,
                                min_sequence_dist=min_sequence_dist,
                                n_pred=n_pred)



    ppv = np.sum(np.tril(full_matrix == 1)) / np.sum(np.tril(np.isfinite(full_matrix)))

    return ppv


def FindNeighbors(edge_site, tmp_matcontact, nbrspin):
    # a function which creates a list containing all the "neighbors" sites
    # i.e. the interacting sites
    
    list_neighbors_site = []
    
    for l in range(0,edge_site):
        if tmp_matcontact[l, edge_site] == 1:
            list_neighbors_site.append(l)
            
    for k in range(edge_site+1, nbrspin):
        if tmp_matcontact[edge_site,k] == 1:
            list_neighbors_site.append(k)
            
    return list_neighbors_site

def FPShortestPath(fp,G, Neighbors_dict):
    length_shortest_path = np.zeros(len(fp))
    fp_pairs = np.linspace(0,len(fp)-1, len(fp))
    shortest_paths = nx.shortest_path(G)
    
    cts_shortest_paths = np.zeros(len(fp))
    for idx, p in enumerate(fp):
        x,y = p

        #except if they are alone sites or if they are in a different component of the graph
        if Neighbors_dict[x] and Neighbors_dict[y] and nx.has_path(G,x,y):
            length_shortest_path[idx] = len(shortest_paths[x][y])
            cts_shortest_paths[idx] = len(list(nx.all_shortest_paths(G,source =  x,target =  y)))
    
    return length_shortest_path, cts_shortest_paths

def GenerateListFP(scoresmat,matcontact,number_contacts):
    assert scoresmat.shape == matcontact.shape
    
    nbrpos = scoresmat.shape[0]
    
    index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(-scoresmat, axis = None), scoresmat.shape)

    N = 0
    TP_coords = []
    FP_coords = []
    all_coords = []
    scores_ordered = []
    for x, y in zip(index_sorted_array_x, index_sorted_array_y):
        if y > x and scoresmat[y,x] != -np.inf:
            N = N + 1
            
            if matcontact[x,y] == 1:

                if N <= number_contacts:
                    TP_coords.append([x,y])
            else:

                if N <= number_contacts:
                    FP_coords.append([x,y])

            all_coords.append([x,y])
            scores_ordered.append(scoresmat[x,y])

    return FP_coords,TP_coords,all_coords,np.array(scores_ordered)



def PPV_FPcharact(path_score,msa_name,path_tociffolder,path_bm_cm_apc,bl_natseq):
    
    scores_matrix = loadscore(path_score, msa_name)

    contact_mask,dist_matrix = ComputeContactMaskDistMat(msa_name,path_tociffolder,path_bm_cm_apc)
    
    pfam_length = len(pfam_to_pdb[msa_name]["pfam"]["seq"])
    n_pred = 2 * pfam_length

    
    assert n_pred == np.sum(contact_mask)
    assert dist_matrix.shape == scores_matrix.shape
    n_residues = dist_matrix.shape[0]
    sequence_proximity_mask = np.abs(
        np.arange(n_residues) - np.arange(n_residues)[:, None]
        ) >= min_sequence_dist

    if bl_natseq:
        eucl_dist_mask = dist_matrix < max_eucl_dist
        contact_mask = np.triu(np.logical_and(eucl_dist_mask, sequence_proximity_mask))       
    
    
    

    scores_matrix = loadscore(path_score, msa_name)
    scores_matrix[sequence_proximity_mask==0] = -np.inf
    fp_coords,tp_coords,allcoords,_ = GenerateListFP(scores_matrix,contact_mask*1,n_pred)

    nbrpos = scores_matrix.shape[0]

    G = nx.from_numpy_matrix(contact_mask*1)
    Neighbors_dict = {site: FindNeighbors(site, contact_mask*1, nbrpos) for site in range(0,nbrpos)}
    

    
    lengthSP, numberSP = FPShortestPath(fp_coords,G, Neighbors_dict)

    return contact_mask,fp_coords,tp_coords,lengthSP,numberSP, n_pred
    


dist_matrix_func = calc_min_dist_matrix  # calc_Calpha_dist_matrix
max_eucl_dist = 8
min_sequence_dist = 5
contact_matrix_kwargs = {"max_eucl_dist": max_eucl_dist,
                         "min_sequence_dist": min_sequence_dist}

pfamids= ['PF00072','PF00512','PF00595','PF02518']
path_folders = './data_example/PPV_MI_plmDCA_scores/'
msa_name_list = ['PF00072','PF00512_full_no_gapped','PF00595_full_no_gapped','PF02518']
path_tociffolder = './data_example/PDB_structures/'
path_bm_cm_apc_folder = './data_example/bmDCA_scores/'


bl_natseq = True
apcstr = ['NOAPC','APC']
phyloweights = ['NOphyloweights','phyloweights']
dataset = ['NAT','bmDCAEQUI','bmDCATREE']
pfamids= ['PF00072','PF00512','PF00595','PF02518']
bl_natseqs = [True,False,False]
#choose the inference method either plmDCA or MI
inf_method = 'plmDCA'
print('Inference done with ',inf_method)
for idxpfam,pfam in enumerate(pfamids):   
    for weights in phyloweights:
        for apc in apcstr:
            for idx_d,d in enumerate(dataset):
                path_score = path_folders+pfam+'/'+inf_method+'/{}_{}_scores_{}_{}.npy'.format(pfam, d,weights,apc)
                path_bm_cm_apc = path_bm_cm_apc_folder+pfam+'_bmDCA_scores_apc.npy'
                cm,fpcoords,tpcoords,lgthsp,nsp,npred = PPV_FPcharact(path_score,msa_name_list[idxpfam],path_tociffolder,path_bm_cm_apc,bl_natseqs[idx_d])
                val,cts = np.unique(lgthsp,return_counts = True)
                print('dataset:',d,' pfam:',pfam,  ' reweighting: ',weights, ' apc: ',apc)
                print('PPV = {:.2f}'.format(len(tpcoords)/(len(tpcoords)+len(fpcoords))), 'Number of indirect fp: {}'.format(cts[val==3]))
                print('number of contacts: ', npred, ' ',len(tpcoords)+len(fpcoords))
                                
                             
                                