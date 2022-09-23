#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute G scores
"""

import numpy as np

from tqdm import tqdm
from math import exp, expm1
from datetime import datetime, date
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25})
import h5py
import os
import networkx as nx
from scipy.stats import entropy

def OpenDataPhylogenyDatasetScanMfixedT(pathdata, pathcontactmat):
    
    file = h5py.File(pathdata,'r')
    chains = file['Chains']

    matrix_contact  = np.array(file['Matrix_contact'])

    temps = np.array(file['Temperatures'])

    matrix_contact = np.load(pathcontactmat)
    allchains = np.array(file['allchains'])

    chains_t = np.array(chains)
    file.close()
    
    return matrix_contact, chains_t,temps,allchains


def OpenJuliaFile(path):
    file = h5py.File(path,'r')
    Jtensor = np.array(file['couplings'])
    
    file.close()

    return Jtensor


def Inference_MF(mat_corr, matrix_contacts,bl_abs):
    """
    Infer the contacts using MF approximation
    """
    val,cts = np.unique(matrix_contacts,return_counts = True)
    nbrcontacts = cts[val == 1]
    
    # inverse of the correlation matrix to get the couplings
    inferred_couplings = np.linalg.inv(mat_corr)

    TP = []

    # order the 2d array and find the index of the sorted values in the matrix
    if bl_abs:
        index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(-np.abs(inferred_couplings), axis=None), inferred_couplings.shape)
        inferred_couplings = -np.abs(inferred_couplings)
    else:
        index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(inferred_couplings, axis=None), inferred_couplings.shape)


    idx_flip = list(index_sorted_array_x)
    idy_flip = list(index_sorted_array_y)


    FP = []
    listFPJij = []
    TP_coords = []
    listTPJij = []
    
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
                    listTPJij.append(-inferred_couplings[x,y])
            else:

                if N <= nbrcontacts:
                    FP.append([x,y])
                    listFPJij.append(-inferred_couplings[x,y])


            list_tp.append(TP)

            all_coords.append([x,y])

            list_tp_fraction_allpairs.append(TP/N)

    return list_tp_fraction_allpairs, FP,listFPJij, TP_coords, listTPJij

def Inference_Correlations(mat_corr, matrix_contacts,bl_abs):
    """
    infer using the correlations between sites. 
    """
    TP = []
    val,cts = np.unique(matrix_contacts,return_counts = True)
    nbrcontacts = cts[val == 1]
    # order the 2d array and find the index of the sorted values in the matrix
    if bl_abs:
        index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(np.abs(mat_corr), axis = None), mat_corr.shape)
        mat_corr = np.abs(mat_corr)
    else:
        index_sorted_array_x, index_sorted_array_y  = np.unravel_index(np.argsort(mat_corr, axis = None), mat_corr.shape)


    idx_flip = np.flip(list(index_sorted_array_x))
    idy_flip = np.flip(list(index_sorted_array_y))


    FP = []
    listFpCij = []
    TP_coords = []
    listTpCij = []
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
                    listTpCij.append(mat_corr[x,y])
            else:
                if N <= nbrcontacts:
                    FP.append([x,y])
                    listFpCij.append(mat_corr[x,y])

            list_tp.append(TP)
            all_coords.append([x,y])


            list_tp_fraction_allpairs.append(TP/N)

    return list_tp_fraction_allpairs, FP,listFpCij, TP_coords,listTpCij

def Inference_PLMDCA(Jscore, matrix_contacts):
    """
    Infer the contacts using plm approximation
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
    
    listFPJij = []
    # §TP_coords = []
    listTPJij = []
    
    
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
                    listTPJij.append(Jscore[x,y])
            else:

                if N <= nbrcontacts:
                    FP.append([x,y])
                    listFPJij.append(Jscore[x,y])


            list_tp.append(TP)

            all_coords.append([x,y])

            list_tp_fraction_allpairs.append(TP/N)

    return list_tp_fraction_allpairs, FP,listFPJij,TP_coords,listTPJij
def zero_sum_gauge(J2):
    J2_zs = J2.copy()
    J2_zs -= np.mean(J2, axis=2, keepdims=True)
    J2_zs -= np.mean(J2, axis=3, keepdims=True)
    J2_zs += np.mean(J2, axis=(2, 3), keepdims=True)
    
    return J2_zs

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


def NewScoreEarliestGenMut(allchains,pairslist,number_generations):
    
    newscore_em = []
    
    for idxfp,fp in enumerate(pairslist):
        s1,s2 = fp
        
        cl1 = allchains[:,s1]
        cl2 = allchains[:,s2]
        
        init_states1= cl1[0]
        init_states2 = cl2[0]
        
        bl_state1 = False
        bl_state2 = False
        stop = False
        
        for k in range(0,number_generations+1):
            for s in range(0,pow(2,k)):
                idx = pow(2,k)-1+s
                if cl1[idx] != cl1[0]:
                    bl_state1 = True
                    if bl_state2:
                        newscore_em.append(k)
                        stop = True
                        break
                if cl2[idx] != cl2[0]:
                    bl_state2 = True
                    if bl_state1:
                        newscore_em.append(k)
                        stop = True
                        break
            if stop:
                break
        if not stop:
            newscore_em.append(number_generations + 1)
            
    return newscore_em


def MutualInformation(MSA, reg):
    """
    MSA: sequences as input
    reg: the pseudocount/regularisation
    computes the mutual information matrix
    """
    if MSA.dtype == "bool":
        MSA = MSA*1
    MSA = (MSA + 1)/2
    
    nbrstates = np.unique(MSA).shape[0]
    nbrseq,nbrpos = MSA.shape
    mi_matrix = np.zeros((nbrpos,nbrpos))

    binst = np.linspace(0,nbrstates, nbrstates+1)
    frequencies = np.zeros([nbrstates, nbrpos])

    for p in range(0,nbrpos):
        cts,val = np.histogram(MSA[:,p],bins = binst)
        frequencies[:,p] = cts/nbrseq
    for i in range(0,nbrpos):
            for j in range(0,nbrpos):
                if i != j:
                    fri = reg/(nbrstates) + (1-reg)*frequencies[:,i]
                    frj = reg/(nbrstates) + (1-reg)*frequencies[:,j]

                    hist2d,_,_ = np.histogram2d(MSA[:,i],MSA[:,j],bins = binst) 
                    jointfreq = (reg/(nbrstates)**2) + (1-reg)*hist2d/nbrseq
                    jointentropy = entropy(jointfreq.flatten())
                    
                    entropy_i = entropy(fri)
                    entropy_j = entropy(frj)


                    mi_matrix[i,j] = (entropy_i + entropy_j - jointentropy)
                else:
                    fri = reg/(nbrstates) + (1-reg)*frequencies[:,i]
                    submatrix = np.zeros([nbrstates, nbrstates])
                    submatrix[np.diag_indices_from(submatrix)] = fri

                    jointentropy = entropy(submatrix.flatten())
                    entropy_i = entropy(fri)
                    
                    mi_matrix[i,j] = (entropy_i + entropy_i - jointentropy)

    return mi_matrix


def ComputeCorrelationMatrix2(mat, pseudocount):
    
    nbr_spins = len(mat[0,:])
    nbr_chains = len(mat[:,0])
    mat = np.array(mat,ndmin = 2, dtype = np.float64)
    average_spin = np.average(mat, axis = 0)[:,None]
    
    directcorr = np.dot(mat.T, mat)
    
    directcorr *= np.true_divide(1, nbr_chains, dtype = np.float64)
    
    correlation_matrix = np.dot(1.0-pseudocount, directcorr) - np.dot(pow(1-pseudocount,2),np.outer(average_spin.T, average_spin)) + np.dot(pseudocount,np.identity(nbr_spins))
    

    return correlation_matrix

def ReturnJAndC(finalchains,pseudocount, bl_abs):
    correlationmatrix = ComputeCorrelationMatrix2(finalchains,pseudocount)
    C = ComputeCorrelationMatrix2(finalchains,0)
    
    inferred_couplings = np.linalg.inv(correlationmatrix)
    if bl_abs:
        inferred_couplings = -np.abs(inferred_couplings)
        C = np.abs(C)
    
    return -inferred_couplings,C
    
def ReturnAllpairscouplings(allpairs,matscore):
    allcouplings = []
    for pair in allpairs:
        x,y = pair
        allcouplings.append(matscore[x,y])
    return allcouplings

def ViolinPlot(earliestmutscore,couplings, T,mu,strfp,strcoupling,number_generations,ylow,yhigh):

    plt.figure()
    seuil = 15
    plt.title(strfp + ' Coupling '+ strcoupling+ ' vs generation earliest mutation \n at T = {}, $\mu$ = {}, threshold = {}'.format(T,mu,seuil))
    couplings = np.array(couplings)
    earliestmutscore = np.array(earliestmutscore)
    positions = np.unique(earliestmutscore)
    
    data = [couplings[earliestmutscore == p] for p in positions if len(couplings[earliestmutscore == p]) > seuil]
    
    median = [np.percentile(d,50) for d in data]
    quantile90 = [np.percentile(d,90) for d in data]
    quantile10 = [np.percentile(d,10) for d in data]
    

    
    datapoints = [couplings[earliestmutscore == p] for p in positions if len(couplings[earliestmutscore == p]) <= seuil]


    positions_pts = [p for p in positions if len(couplings[earliestmutscore == p]) <= seuil ]
    positions_data =  [p for p in positions if len(couplings[earliestmutscore == p]) > seuil ]
    quantileslist = [[0.1,0.5,0.9] for p in positions_data]
    meandata = [np.mean(d) for d in data]
    nbrpts = []
   
    for p in positions:
        nbrpts.append(len(couplings[earliestmutscore == p]))
    plt.violinplot(data,positions_data, vert= True, showextrema=False)

    for idxd,d in enumerate(datapoints):
        plt.plot(positions_pts[idxd]*np.ones(len(d)),d,'o',color = 'C0')
    
    plt.plot(positions_data,quantile90,'x-', label = '90th percentile', markersize = 10)
    plt.plot(positions_data,median,'x-', color = 'red',label = 'Median', markersize = 10)
    plt.plot(positions_data,quantile10,'x-', label = '10th percentile', markersize = 10)

    plt.legend()
    plt.xlabel('Earliest mutation generation')
    plt.ylabel('Inferred Couplings')
    plt.xlim([0.5,number_generations+1.5])
    xtickspos = np.arange(1,number_generations+2,1)
    plt.xticks(xtickspos)



def ConservationPairsListEntropy(listpairs,finalchains):
    conservation =[]
    nbrseq, nbrsites = finalchains.shape
    nbrstates = np.unique(finalchains).shape[0]
    h_entropy = np.zeros(nbrsites)
    for j in range(0,nbrsites):
        val,cts = np.unique(finalchains[:,j],return_counts = True)

        h_entropy[j] = entropy(cts/nbrseq,base = 2)
    conservation_singlesite = 1 - h_entropy

    for pair in listpairs:
        s1,s2 = pair
        
        conservation.append(0.5*(conservation_singlesite[s1]+ conservation_singlesite[s2]))

    return conservation

def APC(mat):
    matcopy = mat.copy()
    mat -= (np.mean(matcopy, axis=1, keepdims=True) * np.mean(matcopy, axis=0, keepdims=True)) / np.mean(matcopy)
    return mat

def AllpairsScore(number_spins,allchains,number_generations,finalchains,pseudocount,bl_abs,Jplm,pc_mi, bl_apc):
    allpairs = []
    for j in range(0,number_spins):
        for i in range(j+1,number_spins):
            allpairs.append([j,i])
    
    earliestmutscoreallpairs = NewScoreEarliestGenMut(allchains, allpairs, number_generations)
    
    J,C = ReturnJAndC(finalchains,pseudocount,bl_abs)
    MI =  MutualInformation(finalchains, pc_mi)
    
    if bl_apc:
        MI = APC(MI)
        J = APC(J)
        C = APC(C)
        Jplm = APC(Jplm)

    
    allcouplingsJ = ReturnAllpairscouplings(allpairs,J)
    allcouplingsC = ReturnAllpairscouplings(allpairs,C)
    allcouplingsJplm = ReturnAllpairscouplings(allpairs,Jplm)
    allcouplingsMI = ReturnAllpairscouplings(allpairs, MI)
    return allcouplingsJ, earliestmutscoreallpairs,allcouplingsC,allpairs,allcouplingsJplm,allcouplingsMI
    


def TracePairTree(listpair, indexpair,allchains, number_generations):
    
    nbr_total_seq, nbr_spins = allchains.shape
    final_nbr_seq = pow(2,number_generations)
    state_matrix = np.zeros([number_generations+1,final_nbr_seq])
    
    #  score 1  if s1,s2 = (1,1)/ 2 if s1,s2 = (-1,1)/ 3 if s1,s2 = (1,-1)( 4 if s1, s2 =(-1,-1)
    s1,s2 = listpair[indexpair]
    cl1 = allchains[:,s1]
    cl2 = allchains[:,s2]
    
    for k in range(0,number_generations+1):
        for s in range(0,pow(2,k)):
            idx = pow(2,k)-1+s
            
            
            if cl1[idx] == 1 and cl2[idx] == 1:
                score = 1
            elif cl1[idx] == -1 and cl2[idx] == 1:
                score = 2
            elif cl1[idx] == 1 and cl2[idx] == -1:
                score = 3
            elif cl1[idx] == -1 and cl2[idx] == -1:
                score = 4
            idxtmp = int(final_nbr_seq/2 - pow(2,k)/2 + s)
            state_matrix[k,idxtmp] = score
    state_matrix[state_matrix == 0] = np.nan
    return state_matrix
    
def PlotPairTree(matrix_state,t,m,g):
    plt.figure(figsize = (15,15))
    plt.title('State of two sites \n T = {}, $\mu$ = {}, G = {}'.format(int(t),int(m),int(g)))
    plt.pcolormesh(matrix_state, norm = None, shading = 'flat',cmap = 'Paired')
    plt.ylabel('Generation')
    plt.xlabel('Sequences')
    plt.gca().invert_yaxis()
    cbar = plt.colorbar()
    cbar.set_ticks([1,2,3,4])
    cbar.set_ticklabels([r'$\uparrow\uparrow$', r'$\downarrow\uparrow$', r'$\uparrow\downarrow$', r'$\downarrow\downarrow$']) 

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
        if Neighbors_dict[x] and Neighbors_dict[y]:
            length_shortest_path[idx] = len(shortest_paths[x][y])
            cts_shortest_paths[idx] = len(list(nx.all_shortest_paths(G,source =  x,target =  y)))
    
    return length_shortest_path, cts_shortest_paths



############################# MAIN ####################################
if __name__ == '__main__':
    #path to the folder that contains an example of sequences to compute the G score, can be changed acccordingly.
    pathtodatafolder = './example/Gscore_computation/sequences/'
    filelist = os.listdir(pathtodatafolder)

    #path to the contact maps used in the paper
    pathcontacmat = './contact_maps/N200p0_02/contactmat_n200p0_02.npy'

    #path to the folder that contains the plmDCA inference of the sequences mentioned above using the plmDCA package, can be changed acccordingly.
    pathtoplmcouplingsfolder = './example/Gscore_computation/plminference/'
    couplingsfilelist = os.listdir(pathtoplmcouplingsfolder)

    pseudocount = 0.5
    pc_mi = 0.01
    bl_abs = True
    bl_apc = False
    number_generations = 11
    indextemperature = 1
    mutation = 5

    ##path where the data is saved, everything is saved in there.
    path_save = './example/save_folder_example/'




    for f in filelist:
        if f.startswith('.'):
            filelist.remove(f)

    for f in couplingsfilelist:
        if f.startswith('.'):
            couplingsfilelist.remove(f)

    allcouplingmf_manyrealisations = []
    earliestmutscoreallpairs_manyrealisations = []
    allcouplingC_manyrealisations = []
    allcouplingMI_manyrealisations = []
    allcouplingJ_manyrealisationsplmdca = []
    conservationallpairs_manyrealisations = []

    nbrfiles =len(filelist)
    from tqdm import tqdm
    for f in tqdm(filelist):
        pathtofile = pathtodatafolder+f

        nbrfile = f[f.find('filenbr')+len('filenbr'):f.find('.')]
        strtofind = 'filenbr' + nbrfile + '.'
        filecoupling = ''


        for ftmp in couplingsfilelist:
            if strtofind in ftmp:
                filecoupling = ftmp

        pathtocoupling = pathtoplmcouplingsfolder + filecoupling

        Jtensor = OpenJuliaFile(pathtocoupling)
        Jtemperature = Jtensor[:,:,:,:,indextemperature]

        Jtemperature = zero_sum_gauge_frob_scores(Jtemperature,bl_apc)

        matrixcontact, chainst, temperatures, allchains = OpenDataPhylogenyDatasetScanMfixedT(pathtofile, pathcontacmat)
        number_spins = matrixcontact.shape[0]

        allcouplingmf, earliestmutscoreallpairs,allcouplingsC,allpairs,allcouplingplm,allcouplingsMI = AllpairsScore(number_spins,allchains[indextemperature,:,:],number_generations,chainst[indextemperature,:,:],pseudocount,bl_abs,Jtemperature,pc_mi,bl_apc)

        allcouplingMI_manyrealisations = allcouplingMI_manyrealisations + allcouplingsMI

        allcouplingmf_manyrealisations = allcouplingmf_manyrealisations +  allcouplingmf

        allcouplingJ_manyrealisationsplmdca = allcouplingJ_manyrealisationsplmdca + allcouplingplm

        allcouplingC_manyrealisations = allcouplingC_manyrealisations + allcouplingsC

        earliestmutscoreallpairs_manyrealisations = earliestmutscoreallpairs_manyrealisations + earliestmutscoreallpairs

        conservationallpairs_manyrealisations = conservationallpairs_manyrealisations + ConservationPairsListEntropy(allpairs, chainst[indextemperature,:,:])


    np.save(path_save + 'CouplingsCORR_Allpairs.npy',allcouplingC_manyrealisations)
    np.save(path_save + 'CouplingMF_Allpairs.npy',allcouplingmf_manyrealisations)
    np.save(path_save + 'CouplingPLM__Allpairs.npy',allcouplingJ_manyrealisationsplmdca)
    np.save(path_save + 'couplingsMI_Allpairs.npy',allcouplingMI_manyrealisations)

    np.save(path_save + 'EMG_scores_abs_apc.npy',earliestmutscoreallpairs_manyrealisations)

    ViolinPlot(earliestmutscoreallpairs_manyrealisations,allcouplingMI_manyrealisations, temperatures[indextemperature],mutation,'Allcouplings Mij {} realisations'.format(nbrfiles),'\n ',number_generations,0.3,0.3)

    ####### EXAMPLE GRAPH PROPERTIES FOR ALL PAIRS IN THE GRAPH ########
    pathcontactmat = './contact_maps/N200p0_02/contactmat_n200p0_02.npy'
    matrix_contact = np.load(pathcontacmat)
    number_spins = matrix_contact.shape[0]
    G = nx.from_numpy_matrix(matrix_contact)
    Neighbors_dict = {site: FindNeighbors(site, matrix_contact, number_spins) for site in range(0,number_spins)}

    allpairs_graph = []
    for j in range(0,number_spins):
        for i in range(j+1,number_spins):
            allpairs_graph.append([j,i])

    lengthSP, numberSP = FPShortestPath(allpairs_graph,G, Neighbors_dict)
    nbrkeys = len(Neighbors_dict.keys())
    listofNN = [len(Neighbors_dict[k]) for k in range(nbrkeys)]
    listofNN = np.array(listofNN)
    print('Shortest path for each pair in the allpairs_graph list', lengthSP)
    print('Number of NN for each site: ',listofNN)
