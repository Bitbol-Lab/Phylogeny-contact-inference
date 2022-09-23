#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer scores from sequences
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from numba import njit,prange,jit
from sklearn import metrics
from scipy.stats import entropy

##other dataset
def OpenDataClusterDataset(pathdata, pathcontactmat):
    
    file = h5py.File(pathdata,'r')
    chains = file['Chains']
    temperatures = np.array(file['Temperatures'])
    matrix_contact = np.load(pathcontactmat)
    chains_t = np.array(chains)
    file.close()
    
    return matrix_contact, chains_t, temperatures


##other dataset
def OpenDataPhylogenyDatasetScanMfixedT(pathdata, pathcontactmat):
    
    file = h5py.File(pathdata,'r')
    chains = file['Chains']

    matrix_contact  = np.array(file['Matrix_contact'])

    matrix_contact = np.load(pathcontactmat)
    mutations = np.array(file['Mutations'])

    chains_t = np.array(chains)
    file.close()
    
    return matrix_contact, chains_t,mutations

def OpenDataPhylogenyDatasetScanTfixedM(pathdata, pathcontactmat):
    
    file = h5py.File(pathdata,'r')
    chains = file['Chains']
    matrix_contact  = np.array(file['Matrix_contact'])
    matrix_contact = np.load(pathcontactmat)
    temperatures = np.array(file['Temperatures'])
    chains_t = np.array(chains)
    file.close()
    
    return matrix_contact, chains_t,temperatures

def ComputeCorrelationMatrix2(mat, pseudocount):
    
    nbr_spins = len(mat[0,:])
    nbr_chains = len(mat[:,0])
    mat = np.array(mat,ndmin = 2, dtype = np.float64)
    average_spin = np.average(mat, axis = 0)[:,None]
    
    directcorr = np.dot(mat.T, mat)

    directcorr *= np.true_divide(1, nbr_chains, dtype = np.float64)
    
    correlation_matrix = np.dot(1.0-pseudocount, directcorr) - np.dot(pow(1-pseudocount,2),np.outer(average_spin.T, average_spin)) + np.dot(pseudocount,np.identity(nbr_spins))


    return correlation_matrix


def Inference_MF(mat_corr, matrix_contacts,bl_abs,bl_apc):
    """
    Infer the contacts using MF approximation
    """
    val,cts = np.unique(matrix_contacts,return_counts = True)
    nbrcontacts = cts[val == 1]
    flag = True
    # inverse of the correlation matrix to get the couplings
    
    try:
        inferred_couplings = np.linalg.inv(mat_corr)
    except:
        flag = False
        
    if flag:
        if bl_abs:
            inferred_couplings = np.abs(inferred_couplings)
            
        if bl_apc:
            np.fill_diagonal(inferred_couplings,0)
            S = inferred_couplings.copy()
            inferred_couplings -= (np.mean(S, axis=1, keepdims=True) * np.mean(S, axis=0, keepdims=True)) / np.mean(S)
    
        
        
        TP = []
    
        # order the 2d array and find the index of the sorted values in the matrix
        if bl_abs:
            index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(-inferred_couplings, axis=None), inferred_couplings.shape)
        else:
            index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(inferred_couplings, axis=None), inferred_couplings.shape)
    
    
        idx_flip = list(index_sorted_array_x)
        idy_flip = list(index_sorted_array_y)
    
        # indirect_corr_second_order =[]
    
        FP = []
    
        TP_coords = []
        all_coords = []
        N = 0 
        number_pairs = []
    
        list_tp = []
        TP = 0
    
        list_tp_fraction_allpairs = []
    
    
        for x, y in zip(idx_flip, idy_flip):
    
            # just look at the elements above the diagonal as symmetric matrix
            # to not count twice each contact
            if y > x:
    
                N = N + 1
    
                number_pairs.append(N)

    
                if matrix_contacts[x,y] == 1:
                    TP = TP + 1
                    if N <= nbrcontacts:
                        TP_coords.append([x,y])
                else:
    
                    if N <= nbrcontacts:
                        FP.append([x,y])

    
                list_tp.append(TP)
    
                all_coords.append([x,y])
    
                list_tp_fraction_allpairs.append(TP/N)
    
        return list_tp_fraction_allpairs, FP
    
    else:
        mat = np.zeros(nbrcontacts)
        mat[:] = np.nan
        return mat,mat


def Inference_Correlations(mat_corr, matrix_contacts,bl_abs,bl_apc):
    """
    infer using the correlations between sites. 
    """
    TP = []
    val,cts = np.unique(matrix_contacts,return_counts = True)
    nbrcontacts = cts[val == 1]
    # order the 2d array and find the index of the sorted values in the matrix

    if bl_abs:
        mat_corr = np.abs(mat_corr)
        
    if bl_apc:
        np.fill_diagonal(mat_corr,0)
        S = mat_corr.copy()
        mat_corr -= (np.mean(S, axis=1, keepdims=True) * np.mean(S, axis=0, keepdims=True)) / np.mean(S)

    index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(mat_corr, axis = None), mat_corr.shape)

    idx_flip = np.flip(list(index_sorted_array_x))
    idy_flip = np.flip(list(index_sorted_array_y))


    FP = []
    TP_coords = []
    all_coords = []


    N = 0 
    number_pairs = []

    list_tp = []
    TP = 0

    list_tp_fraction_allpairs = []


    for x, y in zip(idx_flip, idy_flip):

        # just look at the elements above the diagonal as symmetric matrix
        # to not count twice each contact
        if y > x:
            N = N + 1
            number_pairs.append(N)

            if matrix_contacts[x,y] == 1:
                TP = TP + 1
                if N <= nbrcontacts:
                    TP_coords.append([x,y])
            else:
                if N <= nbrcontacts:
                    FP.append([x,y])


            list_tp.append(TP)
            all_coords.append([x,y])


            list_tp_fraction_allpairs.append(TP/N)

    return list_tp_fraction_allpairs, FP

def InferFractions(chains,matcontact, pseudocount, bl_abs,bl_apc):
    corrmatrix = ComputeCorrelationMatrix2(chains,pseudocount)
    
    corrmatrix_a0 = ComputeCorrelationMatrix2(chains,0)
    
    listmf,_ = Inference_MF(corrmatrix, matcontact,bl_abs,bl_apc)
    listcorr,_ = Inference_Correlations(corrmatrix_a0, matcontact,bl_abs,bl_apc)
    return listmf, listcorr


def TPfractionATNcontact(chains,matcontact, pseudocount,bl_abs,bl_apc):
    
    matcontact_temp = matcontact + 1
    off_diag = np.triu(matcontact_temp, k = 1)
    val,cts = np.unique(off_diag,return_counts = True)
    nbr_contacts = cts[val == 2]
    nbr_contacts = int(nbr_contacts[0])

    
    listmf,listcorr = InferFractions(chains, matcontact, pseudocount,bl_abs,bl_apc)
    
    return listmf[nbr_contacts-1],listcorr[nbr_contacts-1]


def TPfractionNcontactvsTemperature(chains,temperatures,matcontact,pseudocount,bl_abs,bl_apc):
    tpfrac_mf_alltemps = np.zeros(len(temperatures))
    
    tpfrac_corr_alltemps = np.zeros(len(temperatures))
    
    for idxt,t in enumerate(temperatures):
        tpfrac_mf,tpfrac_corr = TPfractionATNcontact(chains[idxt,:,:], matcontact, pseudocount,bl_abs,bl_apc)
        
        tpfrac_mf_alltemps[idxt] = tpfrac_mf
        tpfrac_corr_alltemps[idxt] = tpfrac_corr
        
    return tpfrac_mf_alltemps, tpfrac_corr_alltemps


def InferenceMF_Rates(mat_corr, matrix_contacts,nbr_contacts, nbr_noncontacts,bl_abs,bl_apc):

    flag = True
    # inverse of the correlation matrix to get the couplings
    
    try:
        inferred_couplings = np.linalg.inv(mat_corr)
    except:
        flag = False
    
    
    if flag:
    
        
        
        if bl_abs:
            inferred_couplings = np.abs(inferred_couplings)
            
        if bl_apc:
            S = inferred_couplings.copy()
            inferred_couplings -= (np.mean(S, axis=1, keepdims=True) * np.mean(S, axis=0, keepdims=True)) / np.mean(S)
    
        
        
        TP = []
    
        # order the 2d array and find the index of the sorted values in the matrix
        if bl_abs:
            index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(-inferred_couplings, axis=None), inferred_couplings.shape)
        else:
            index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(inferred_couplings, axis=None), inferred_couplings.shape)
    
    
        idx_flip = list(index_sorted_array_x)
        idy_flip = list(index_sorted_array_y)
        
        TP_rate = []
        FP_rate = []
        
        ctsTP = 0
        ctsFP = 0
        
        for x, y in zip(idx_flip, idy_flip):
            
            if y > x:
                if matrix_contacts[x,y] == 1:
                    ctsTP += 1
                else:
                    ctsFP += 1
                
                TP_rate.append(ctsTP/nbr_contacts)
                FP_rate.append(ctsFP/nbr_noncontacts)
                
        return TP_rate, FP_rate
    else:
        tprate = np.zeros(nbr_contacts+nbr_noncontacts)
        tprate[:] = np.nan

        return tprate, tprate        

def InferenceCorr_Rates(mat_corr, matrix_contacts,nbr_contacts,nbr_noncontacts, bl_abs,bl_apc):
    

    if bl_abs:
        mat_corr = np.abs(mat_corr)
        
    if bl_apc:
        S = mat_corr.copy()
        mat_corr -= (np.mean(S, axis=1, keepdims=True) * np.mean(S, axis=0, keepdims=True)) / np.mean(S)


    index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(mat_corr, axis = None), mat_corr.shape)

    
    
    idx_flip = np.flip(list(index_sorted_array_x))
    idy_flip = np.flip(list(index_sorted_array_y))
    
    TP_rate = []
    FP_rate = []
    
    ctsTP = 0
    ctsFP = 0
    
    for x, y in zip(idx_flip, idy_flip):
        
        if y > x:
            if matrix_contacts[x,y] == 1:
                ctsTP += 1
            else:
                ctsFP += 1
            
            TP_rate.append(ctsTP/nbr_contacts)
            FP_rate.append(ctsFP/nbr_noncontacts)
            
    return TP_rate, FP_rate

def ComputeRates(chains,matcontact, pseudocount, bl_abs,bl_apc):
    matcontact_temp = matcontact + 1
    off_diag = np.triu(matcontact_temp, k = 1)
    val,cts = np.unique(off_diag,return_counts = True)
    nbr_contacts = cts[val == 2]
    nbr_noncontacts = cts[val == 1]
    
    corrmatrix = ComputeCorrelationMatrix2(chains,pseudocount)
    corrmatrix_a0 = ComputeCorrelationMatrix2(chains,0)
    
    TPrate_mf,FPrate_mf = InferenceMF_Rates(corrmatrix, matcontact, nbr_contacts, nbr_noncontacts,bl_abs,bl_apc)
    TPrate_corr, FPrate_corr = InferenceCorr_Rates(corrmatrix_a0, matcontact, nbr_contacts, nbr_noncontacts, bl_abs,bl_apc)
    
    return TPrate_mf,FPrate_mf,TPrate_corr, FPrate_corr

def ComputeAUC(tpr,fpr):
    return metrics.auc(fpr, tpr)

def ComputeAUCOneparameter(chains,temperatures,matcontact,pseudocount, bl_abs,bl_apc):
    
    auc_mf =  np.zeros(len(temperatures))
    auc_corr =  np.zeros(len(temperatures))
    
    for idxt,t in enumerate(temperatures):
        tpr_mf,fpr_mf,tpr_corr,fpr_corr = ComputeRates(chains[idxt,:,:], matcontact, pseudocount, bl_abs,bl_apc)
        auc_mf[idxt] = metrics.auc(fpr_mf, tpr_mf) 
        auc_corr[idxt] = metrics.auc(fpr_corr, tpr_corr)
    
    return auc_mf, auc_corr

def RandomROC(matcontact):
    nbrsites = matcontact.shape[0]

    matcontact_temp = matcontact + 1
    off_diag = np.triu(matcontact_temp, k = 1)
    val,cts = np.unique(off_diag,return_counts = True)
    nbr_contacts = cts[val == 2]
    nbr_noncontacts = cts[val == 1]
    
    list_pairs = []
    for i in range(0,nbrsites):
        for j in range(i+1,nbrsites):
            list_pairs.append([i,j])
    
    indexes = list(np.linspace(0,len(list_pairs)-1,len(list_pairs),dtype = np.int))
    shuffled_indexes = np.random.choice(indexes,len(indexes),replace = False)
    
    TPrate = []
    FPrate = []
    ctsTP = 0
    ctsFP = 0
    
    for ind in list(shuffled_indexes):
        x,y = list_pairs[ind]
        if y > x:
            if matcontact[x,y] == 1:
                ctsTP += 1
            else:
                ctsFP += 1
            
            TPrate.append(ctsTP/nbr_contacts)
            FPrate.append(ctsFP/nbr_noncontacts)

    return TPrate, FPrate

#############PLM DCA PART ####################################################

def InferencePLMDCA_Rates(Jscore, matrix_contacts,nbr_contacts, nbr_noncontacts):
    

    # order the 2d array and find the index of the sorted values in the matrix

    index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(-Jscore, axis=None), Jscore.shape)

    idx_flip = list(index_sorted_array_x)
    idy_flip = list(index_sorted_array_y)
    
    TP_rate = []
    FP_rate = []
    
    ctsTP = 0
    ctsFP = 0
    
    for x, y in zip(idx_flip, idy_flip):
        
        if y > x:
            if matrix_contacts[x,y] == 1:
                ctsTP += 1
            else:
                ctsFP += 1
            
            TP_rate.append(ctsTP/nbr_contacts)
            FP_rate.append(ctsFP/nbr_noncontacts)
            
    return TP_rate, FP_rate

def Inference_PLMDCA(Jscore, matrix_contacts):
    """
    Infer the contacts using MF approximation
    """
    val,cts = np.unique(matrix_contacts,return_counts = True)
    nbrcontacts = cts[val == 1]
    
    # inverse of the correlation matrix to get the couplings
    # inferred_couplings = np.linalg.inv(mat_corr)

    TP = []

    # order the 2d array and find the index of the sorted values in the matrix
    index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(-Jscore, axis=None), Jscore.shape)


    idx_flip = list(index_sorted_array_x)
    idy_flip = list(index_sorted_array_y)


    FP = []

    TP_coords = []
    all_coords = []
    N = 0 
    number_pairs = []

    list_tp = []
    TP = 0

    list_tp_fraction_allpairs = []


    for x, y in zip(idx_flip, idy_flip):

        # just look at the elements above the diagonal as symmetric matrix
        # to not count twice each contact
        if y > x:
            N = N + 1
            number_pairs.append(N)

            if matrix_contacts[x,y] == 1:
                TP = TP + 1
                if N <= nbrcontacts:
                    TP_coords.append([x,y])
            else:

                if N <= nbrcontacts:
                    FP.append([x,y])

            list_tp.append(TP)

            all_coords.append([x,y])

            list_tp_fraction_allpairs.append(TP/N)

    return list_tp_fraction_allpairs, FP



def OpenJuliaFile(path):
    file = h5py.File(path,'r')
    Jtensor = np.array(file['couplings'])
    
    file.close()

    return Jtensor

def OpenJuliaFileWithParam(path):
    file = h5py.File(path,'r')
    Jtensor = np.array(file['couplings'])
    regparam = np.array(file['regularisationparam'])
    file.close()

    return Jtensor, regparam
        


def zero_sum_gauge_frob_scores(J2, apc=True):
    """Compute a score for contacts between sites by first passing to a zero-sum gauge, then computing a Frobenius
    norm, and finally applying the average product correction (APC) if desired."""
    # Pass to zero-sum gauge
    J2_zs = J2.copy()
    J2_zs -= np.mean(J2, axis=2, keepdims=True)
    J2_zs -= np.mean(J2, axis=3, keepdims=True)
    J2_zs += np.mean(J2, axis=(2, 3), keepdims=True)

    # Frobenius norm
    S_frob = np.linalg.norm(J2_zs, axis=(2, 3), ord='fro')

    # Average-product correction
    S = S_frob
    if apc:
        S -= (np.mean(S_frob, axis=1, keepdims=True) * np.mean(S_frob, axis=0, keepdims=True)) / np.mean(S_frob)

    return S

def zero_sum_gauge(J2):
    J2_zs = J2.copy()
    J2_zs -= np.mean(J2, axis=2, keepdims=True)
    J2_zs -= np.mean(J2, axis=3, keepdims=True)
    J2_zs += np.mean(J2, axis=(2, 3), keepdims=True)
    
    return J2_zs

def InferPLMDCAVSParameter(Jtensor, parameterslist, matcontact, blfrob,bl_apc):
    assert Jtensor.shape[-1] == len(parameterslist)
    
    
    tpfraction_plmdca = np.zeros(len(parameterslist))
    auc_plmdca =  np.zeros(len(parameterslist))
    
    
    matcontact_temp = matcontact + 1
    off_diag = np.triu(matcontact_temp, k = 1)
    val,cts = np.unique(off_diag,return_counts = True)
    nbr_contacts = cts[val == 2]
    nbr_noncontacts = cts[val == 1]
    nbr_contacts = int(nbr_contacts[0])
    
    
    for idxp, p in enumerate(parameterslist):
        
        if blfrob:
            Jscore = zero_sum_gauge_frob_scores(Jtensor[:,:,:,:,idxp], bl_apc)
        else:
            print('NO APC for plm -> change if needed')
            Jscoretmp = zero_sum_gauge(Jtensor[:,:,:,:,idxp])
            Jscore = Jscoretmp[:,:,0,0]
        
        tprate,fprate = InferencePLMDCA_Rates(Jscore, matcontact, nbr_contacts, nbr_noncontacts)
        
        auc_plmdca[idxp] = metrics.auc(fprate, tprate) 
        list_tpfraction,_ = Inference_PLMDCA(Jscore, matcontact)
        tpfraction_plmdca[idxp] = list_tpfraction[nbr_contacts-1]
        
    return auc_plmdca, tpfraction_plmdca

def InferPLMVsParam(pathtoplmdca, proba, nbrseqtoselect,matrix_contact):
    
    Jtensor,regparam = OpenJuliaFileWithParam(pathtoplmdca)
    auc_plmfrob,tpfraction_plmfrob = InferPLMDCAVSParameter(Jtensor, regparam, matrix_contact, True)

    auc_plm,tpfraction_plm = InferPLMDCAVSParameter(Jtensor, regparam, matrix_contact, False)
    
    return auc_plmfrob,tpfraction_plmfrob,auc_plm,tpfraction_plm



def InvertSpinsDirection(msa):
    nbrparam,nbrseq,nbrpos = msa.shape
    newmsa = np.empty(msa.shape,dtype = msa.dtype)
    for p in range(0,nbrparam):
        for s in range(0,nbrseq):
            seqtmp = msa[p,s,:]
            if np.sum(seqtmp) >= 0:
                newmsa[p,s,:] = seqtmp
            else:
                newmsa[p,s,:] = (-1)*seqtmp
            
    return newmsa

####################### MUTUAL INFORMATION####################################
def FrequencesJointFrequencies(MSA,reg,frequencies):
    if MSA.dtype == "bool":
        MSA = MSA*1
    MSA = (MSA + 1)/2
    nbrstates = np.unique(MSA).shape[0]
    nbrseq,nbrpos = MSA.shape
    binst = np.linspace(0,nbrstates, nbrstates+1)
    jointfrequencies = np.zeros((nbrpos,nbrpos,nbrstates,nbrstates))
    
    for i in range(0,nbrpos):
        for j in range(i,nbrpos):
            if i != j:
              
                hist2d,_,_ = np.histogram2d(MSA[:,i],MSA[:,j],bins = binst) 

                jointfrequencies[i,j,:,:] = hist2d/nbrseq

            else:

                submatrix = np.zeros([nbrstates, nbrstates])
                submatrix[np.diag_indices_from(submatrix)] = frequencies[:,i]
                jointfrequencies[i, j,:,:] = submatrix 
    
    return jointfrequencies


def ComputeFrequenciesQstates(msa,reg):
    if msa.dtype == "bool":
        msa = msa*1
    msa = (msa + 1)/2
    nbrseq,nbrpos = msa.shape
    l = list(np.unique(msa))
    nbrstates = len(l)
    binst = np.linspace(0,nbrstates,nbrstates+1)
    frequencies = np.zeros([nbrstates, nbrpos])
    entropies = np.zeros(nbrpos)
    for p in range(0,nbrpos):

        cts,val = np.histogram(msa[:,p],bins = binst)

        frequencies[:,p] = cts/nbrseq
        
        fr = reg/(nbrstates) + (1-reg)*frequencies[:,p]
        entropies[p] = entropy(fr)

    return frequencies,entropies

def NewMutualInformation(jointfrequencies,frequencies,reg,entropy_col):
    nbrstates,nbrpos = frequencies.shape

    mi_matrix = np.zeros((nbrpos,nbrpos))
    for i in range(0,nbrpos):
        for j in range(i,nbrpos):
            if i != j:

                jointfreq = (reg/(nbrstates)**2) + (1-reg)*jointfrequencies[i,j,:,:]         
                jointentropy = entropy(jointfreq.flatten())
                
                entropy_i = entropy_col[i]
                entropy_j = entropy_col[j]
                

                mi_matrix[i,j] = (entropy_i + entropy_j - jointentropy)
            else:
                fri = reg/(nbrstates) + (1-reg)*frequencies[:,i]
                submatrix = np.zeros([nbrstates, nbrstates])
                submatrix[np.diag_indices_from(submatrix)] = fri

                jointentropy = entropy(submatrix.flatten())
                entropy_i = entropy_col[i]
                
                mi_matrix[i,j] = (entropy_i + entropy_i - jointentropy)
                
    return np.triu(mi_matrix) + np.tril(mi_matrix.T, -1)


def AUCANDTPMI(chains,temperatures,matcontact,pseudocount_mi,bl_abs,bl_apc):
    
    tpfrac_MI_alltemps = np.zeros(len(temperatures))
    auc_mi =  np.zeros(len(temperatures))
    
    matcontact_temp = matcontact + 1
    off_diag = np.triu(matcontact_temp, k = 1)
    val,cts = np.unique(off_diag,return_counts = True)
    nbr_contacts = int(cts[val == 2])
    nbr_noncontacts = cts[val == 1]
    
    for idxt,t in enumerate(temperatures):

        singlefreq,colentropies = ComputeFrequenciesQstates(chains[idxt,:,:],pseudocount_mi)
        jointfrequencies = FrequencesJointFrequencies(chains[idxt,:,:],pseudocount_mi,singlefreq)
        newMImatrix = NewMutualInformation(jointfrequencies,singlefreq,pseudocount_mi,colentropies)

        listMI,_ = Inference_Correlations(newMImatrix, matcontact,bl_abs,bl_apc)
        
        tpfrac_MI_alltemps[idxt] = listMI[nbr_contacts-1]
        
        tpr_mi, fpr_mi = InferenceCorr_Rates(newMImatrix, matcontact, nbr_contacts, nbr_noncontacts, bl_abs,bl_apc)
        
        auc_mi[idxt] = metrics.auc(fpr_mi, tpr_mi) 
       
    
    return tpfrac_MI_alltemps,auc_mi

#########################MAIN#####################################
if __name__ == '__main__':
    #this path is the path of the folder containing the example sequences on which the inference is performed. It should be changed accordingly.
    pathtofolder = './example/phylogeny/mutations_1_50_temperature_5_generations_11_number_spins_200_starteqchain_p0_02/'
    #path to the contact maps used in the paper.
    pathtocontactmap = './contact_maps/N200p0_02/contactmat_n200p0_02.npy'

    #path to the folder containing the example plmDCA couplings saved in the .jld format and the key 'couplings' should be used when saving the coupling matrix (plmDCA package)
    #or else change the key name in the function OpenJuliaFile
    pathtoplmcouplings = './example/phylogeny/plmcoupling/'

    pseudocount = 0.5
    pseudocount_mi = 0.01

    bl_abs = True
    bl_frob = True
    bl_apc = True

    filelist = os.listdir(pathtofolder)
    for f in filelist:
        if f.startswith('.'):
            filelist.remove(f)


    couplingsfilelist = os.listdir(pathtoplmcouplings)
    for f in couplingsfilelist:
        if f.startswith('.'):
            couplingsfilelist.remove(f)


    pathdata = pathtofolder + filelist[0]

    ###Choose the correct function

    ###phylogeny dataset where temperature is fixed and the number of mutations is varied
    matrix_contact, _, parameters = OpenDataPhylogenyDatasetScanMfixedT(pathdata,pathtocontactmap)
    ###phylogeny dataset where the number of mutations is fixed and the temperature is varied
    # matrix_contact,_,parameters = OpenDataPhylogenyDatasetScanTfixedM(pathdata, pathtocontactmap)
    ###this is opens an equilibrium dataset generated with the cluster
    # matrix_contact,_,parameters = OpenDataClusterDataset(pathdata, pathtocontactmap)

    tpmf_manyrealisations = []
    tpcorr_manyrealisations =[]
    tpplm_manyrealisations = []

    AUCmf_manyrealisations = []
    AUCcorr_manyrealisations =[]
    AUCplm_manyrealisations = []

    TPMI_manyrealisations = []
    AUCMI_manyrealisations = []

    from tqdm import tqdm
    for f in tqdm(filelist):
        pathtofile = pathtofolder + f

        nbrfile = f[f.find('filenbr')+len('filenbr'):f.find('.')]
        strtofind = 'filenbr' + nbrfile + '.'
        filecoupling = ''

        for ftmp in couplingsfilelist:
            if strtofind in ftmp:
                filecoupling = ftmp

        pathtocoupling = pathtoplmcouplings + filecoupling

        Jtensor = OpenJuliaFile(pathtocoupling)




        _, chains_parameters, _ = OpenDataPhylogenyDatasetScanMfixedT(pathtofile,pathtocontactmap)

        # newmsa = InvertSpinsDirection(chains_parameters)


        tpfraction_mf, tpfraction_corr = TPfractionNcontactvsTemperature(chains_parameters,parameters,matrix_contact,pseudocount,bl_abs,bl_apc)
        auc_mf,auc_corr = ComputeAUCOneparameter(chains_parameters, parameters, matrix_contact, pseudocount,bl_abs,bl_apc)

        tpmi,aucmi = AUCANDTPMI(chains_parameters,parameters,matrix_contact,pseudocount_mi,False,bl_apc)

        TPMI_manyrealisations.append(tpmi)
        AUCMI_manyrealisations.append(aucmi)    

        auc_plm,tpfraction_plm = InferPLMDCAVSParameter(Jtensor, parameters, matrix_contact, bl_frob,bl_apc)

        tpmf_manyrealisations.append(tpfraction_mf)
        tpcorr_manyrealisations.append(tpfraction_corr)

        AUCmf_manyrealisations.append(auc_mf)
        AUCcorr_manyrealisations.append(auc_corr)

        AUCplm_manyrealisations.append(auc_plm)
        tpplm_manyrealisations.append(tpfraction_plm)


    ###path of the folder where the results are saved
    path_tosave= './example/save_folder_example/'

    np.save(path_tosave+'tpfrac_mf.npy',tpmf_manyrealisations)
    np.save(path_tosave+'tpfrac_corr.npy',tpcorr_manyrealisations)
    np.save(path_tosave+'tpfrac_plm.npy',tpplm_manyrealisations)
    np.save(path_tosave+'tpfrac_mi.npy',TPMI_manyrealisations)

    np.save(path_tosave+'auc_mf.npy',AUCmf_manyrealisations)
    np.save(path_tosave+'auc_corr.npy',AUCcorr_manyrealisations)
    np.save(path_tosave+'auc_plm.npy',AUCplm_manyrealisations)
    np.save(path_tosave+'auc_mi.npy',AUCMI_manyrealisations)




