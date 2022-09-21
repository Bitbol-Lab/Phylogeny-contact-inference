#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse realistic sequences
"""
import numpy as np
from math import exp, expm1
from datetime import datetime, date
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle

def OpenDataNew(msa_name,key, pathscore,pfam_to_pdb, path_folder_contactmaps):

    
    with open(path_folder_contactmaps+'bmDCA_contactmaps_npred2L_8A.pkl', 'rb') as f:
        bmcontactmaps = pickle.load(f)
    
    pdbcontactmap = np.load(path_folder_contactmaps + msa_name +'_PDB_contact_mask_8A_Allcontacts.npy')
    seq_prox_mask = np.load(path_folder_contactmaps + msa_name + '_seq_prox_mask_min5A.npy')

    pfam_idxs = pfam_to_pdb[msa_name]["pfam"]["idxs"]
    mat = np.load(pathscore)
    scores = mat[np.asarray(pfam_idxs)][:, np.asarray(pfam_idxs)]
    
    return scores,bmcontactmaps[msa_name]["contact_mask"],pdbcontactmap,seq_prox_mask
    
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

def LoadTree(pathpkl):
    with open(pathpkl, 'rb') as f:
        x = pickle.load(f)
    
    return x


def rec_sum_branch_lengths(parent):
    if len(parent.clades):
        for clade in parent.clades:
            clade.branch_length += parent.branch_length
            rec_sum_branch_lengths(clade)
    else:
        return None

def FPanalysisPhyloData(path_topkl):
    tree = LoadTree(path_topkl)
    rec_sum_branch_lengths(tree.clade)
    
    return tree

from tqdm import tqdm

def DictionarySequences(parent,dict_mut,size_seq):
    if len(parent.clades):
        for clade in parent.clades:
            nbr_mut = int(np.rint( clade.branch_length * size_seq))
            if nbr_mut in dict_mut:
                dict_mut[nbr_mut].append(clade.comment)
            else:
                dict_mut[nbr_mut] = [clade.comment]
            DictionarySequences(clade, dict_mut, size_seq)
    else:

        return None

def ComputeGscore(listpairs,dict_seq, root_seq):
    list_n = list(dict_seq.keys())
    list_n.sort()
    gscores = np.zeros(len(listpairs))
    for idp,pair in enumerate(listpairs):
        i,j = pair
        statei = root_seq[i]
        statej = root_seq[j]
        bli = False
        blj = False
        for n in list_n:
            for seq in dict_seq[n]:
                if seq[i] != statei:
                    bli = True
                if seq[j] != statej:
                    blj= True
                
                if bli and blj:
                    gscores[idp] = n
                    break
            if bli and blj:
                break
        
    return gscores
    
def ListScoresFromList(listpairs,mat_scores):
    list_scores = []
    
    for p in listpairs:
        x,y = p
        list_scores.append(mat_scores[x,y])
        
    return list_scores
  
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
                                    "idxs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}}}


####################### MAIN #######################
##path to the sequences dataset generated with on the inferred tree. Here is an example dataset based on the PF72 family.
path_tree = './data_example/Tree_data/sequences/PF00072_0.pkl'
tree = FPanalysisPhyloData(path_tree)
size_seq = int(tree.clade.comment.shape[0])
dict_seq = {}
DictionarySequences(tree.clade,dict_seq,size_seq)
##path folder to the contacts maps. Here is the example to the experimental one of PF72 and the bmDCA inferred contact map of the same family.
path_folder_contactmaps = './data_example/Tree_data/contactmaps/'
keys = ["", "bmDCA", "bmDCA_FastTree_eq"]
msaname = 'PF00072'
##path to the example score, here the score is inferred by MI on the tree dataset of PF72.
path_score = './data_example/Tree_data/scores/PF00072_msa_phylo_equi_bmdca_tree_0_MI_scores.npy'
scores,bm_cm, pdb_cm, seq_prox_mask = OpenDataNew(msaname, keys[0],path_score,pfam_to_pdb,path_folder_contactmaps)
number_contacts_bm = np.sum(bm_cm)
scores[seq_prox_mask == 0] = -np.inf
fpcoords,tpcoords,allcoords,scoreslist =  GenerateListFP(scores,bm_cm*1,number_contacts_bm)
gscores = ComputeGscore(allcoords,dict_seq, tree.clade.comment)
couplings = ListScoresFromList(allcoords,scores)
