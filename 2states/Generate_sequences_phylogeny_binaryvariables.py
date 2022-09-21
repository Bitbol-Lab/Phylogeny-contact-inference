#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic sequences with binary variables with phylogeny
"""
import numpy as np
import random
from tqdm import tqdm
from math import exp, expm1
from datetime import datetime, date
import h5py
import os

startTime = datetime.now()
today = date.today()


def Generate_Graph(Nspin, p):

    #generate a matrix of size len(N)xlen(N), which will return the coupled
    #pairs. The coupled pairs are represented by the indices of the matrix,
    #the entry of the matrix for the coresponding pair is 1 for coupled pairs or
    #0 if they are not linked.
    mat = np.zeros([Nspin, Nspin])

    #iterate over the upper diagonal (diagonal not included)
    #i.e. we iterate over the pairs of the chain.
    for i in range(0,Nspin):
        for j in range(i+1, Nspin):
            #generate random number uniformly between 0 and 1, to compare it
            #with p.
            randnumber = random.uniform(0,1)
            #if the random number is above p than accept the link, i.e. matrix
            #entry is set to 1, otherwise it is set to 0.
            if randnumber < p:
                mat[i,j] = 1
            else:
                mat[i,j] = 0
    # return the final matrix with the couplings.
    return mat

def Load_eq_sequence(pathtodata,temperature):
    """
    Load equilibrium sequence to use it as ancestor sequence
    needs a path to a hdf file containing sequences at equilibrium. 
    The temperature 
    """


    # print(filename)
    file = h5py.File(pathtodata,'r')
    data_cluster = file['Chains']
    nbr_t,nbrchains,nbrspins = data_cluster.shape
    indexchain = np.random.randint(0,nbrchains)
    temperatures = list(file['Temperatures'])
    indexT = temperatures.index(temperature)
    ch = np.array(data_cluster[indexT,indexchain,:], dtype = np.int8)

    
    file.close()
    
    
    return ch


def Mutate(chain, max_nbr_mut, Temp, matcontact):
    """
    Perform a certain number of mutations on a sequence according to the MC 
    step, at fixed temperature.

    Parameters
    ----------
    chain : 1darray
        sequence.
    max_nbr_mut : int
        number of mutations.
    Temp : float
        sampling temperature, "selection strength".
    matcontact : 2darray
        Adjacency matrix of the contact map.

    Returns
    -------
    chain : 1d
        protein sequence after mutations.
    mutation_list : List
        Stores the sites which mutated.

    """
    count_mut = 0
    energy = (-1)*np.matmul(chain,np.matmul(matcontact,chain))
    mutation_list = np.zeros(max_nbr_mut)
    while(count_mut < max_nbr_mut):
        newchain = chain.copy()
        rand_site1 = random.randrange(0,len(chain),1)
        newchain[rand_site1] = newchain[rand_site1]*-1
        new_energ = (-1)*np.matmul(newchain,np.matmul(matcontact,newchain))
        if new_energ < energy:
            #if new config has lower energy, than the new config is accepted
            #and the new values are set.
            
            energy = new_energ
            chain = newchain.copy()
            mutation_list[count_mut] = rand_site1
            count_mut = count_mut + 1

        else:

            #if the new configuration has higher energy than generate a random
            #number uniformly between 0 and 1 and compute the exponential of the
            #energy difference.

            #if the randomly chosen number is lower than the probability given by
            #the exp than accept the new configuration.

            if random.uniform(0,1) < exp((energy-new_energ)/Temp):
                energy = new_energ
                chain = newchain.copy()
                mutation_list[count_mut] = rand_site1
                #a flip has been accepted so add 1 to the total number if acepeted flips.
                count_mut = count_mut + 1


    return chain,mutation_list

def Generate_tree(nbr_gen):
    tree = {}
    for g in range(0,nbr_gen+1):
        list_children = np.linspace(1,pow(2,g),pow(2,g), dtype = np.int16)
        for child in list(list_children):
            tree['{}/{}'.format(g,child)] = None
    return tree



########################### MAIN ###########################

##path to the set of sequences at equilibrium, needed for the root of the phylogeny. Here an example is taken for the input but can be changed
##by generating another dataset of equilibrium sequences.
path_to_eq_seq = './example/equilibrium/2022_09_14_12_24_42_Nspins200_probagraph0_02_flips300_Nchains2048_seed_17_filenbr0.h5'

#number of realisations 
number_averages = 10

for nbrf in tqdm(range(0,number_averages)):
    ##number of mutations
    # mutations_list = list(np.linspace(1,50,50, dtype = np.int))
    mutations_list = [5,10]
    #sampling temperature (should be the same as the one used to generate the ancestor equilibrium sequence)
    Temperature = 5
    #number of generations in the phylogeny
    number_generations = 11
    #number of spins/sites in the graph
    number_spins = 200
    seed = 2*(nbrf+1)
    random.seed(seed)
    #path to the contact map, here is the one used in the paper. It is the same map that generated the equilibrium ancestor sequence.
    matcontact =  np.load('./contact_maps/N200p0_02/contactmat_n200p0_02.npy')
    #the erdos-renyi probability, to the save the parameter value.
    proba = 0.02
    
    final_chains = np.zeros((len(mutations_list),pow(2,number_generations),number_spins),dtype = np.int8)
   
    # if you want to save every sequence in the phylogeny, change the boolean 
    # variable below. Useful for the G score computation.
    save_chainsandmutations = False
    if save_chainsandmutations:
        all_chains = np.zeros((len(mutations_list),2*pow(2,number_generations)-1,number_spins))  
        savedmutations_list = np.zeros((len(mutations_list),2*pow(2,number_generations)-2, number_mutations))
        
    for idxm, mut in enumerate(mutations_list):
            date = today.strftime("%Y_%m_%d_")
            hour = startTime.strftime('%H_%M_%S_')
            number_mutations = mut
            Tree = Generate_tree(number_generations)
            starting_chain = np.zeros(number_spins, dtype = np.int8)
            # start from equilibrium chain
            starting_chain = Load_eq_sequence(path_to_eq_seq, 4.0)

            Tree['0/1'] = starting_chain
            
            if save_chainsandmutations:
                all_chains[idxm,0,:] = starting_chain
                
                ct = 0
            for g in range(1,number_generations+1):
                
                list_parents = np.linspace(1,pow(2,g-1),pow(2,g-1), dtype = np.int16)
                # print('Generation',g)
                for parent in list_parents:
                    # print('parent',parent)
                    chain = Tree['{}/{}'.format(g-1,parent)]
                    newchain1,ml1 = Mutate(chain.copy(), number_mutations, Temperature, matcontact)
                    newchain2,ml2 = Mutate(chain.copy(), number_mutations, Temperature, matcontact)

                    if save_chainsandmutations:
                        all_chains[idxm,ct+1,:] = newchain1
                        all_chains[idxm,ct+2,:] = newchain2
                        savedmutations_list[idxm,ct,:] = ml1[:]
                        savedmutations_list[idxm,ct+1,:] = ml2[:]
                        ct = ct+2
                    
                    Tree['{}/{}'.format(g,2*parent-1)] = newchain1
                    Tree['{}/{}'.format(g,2*parent)] = newchain2
                    
    
            for index_chain, child in enumerate(list(np.linspace(1,pow(2,number_generations),pow(2,number_generations),dtype = np.int16))):
                final_chains[idxm,index_chain,:] = Tree['{}/{}'.format(number_generations,child)]
                        
    ##########SAVE CHAINS, CREATES FOLDER IF NOT EXISTING FOR THE PARAMETERSs#######################
    
    path_store_data = './example/save_folder_example/'
    folderpath = path_store_data + 'mutations_{}_{}_temperature_{}_generations_{}_number_spins_{}_starteqchain_p{}/'.format(int(min(mutations_list)),int(max(mutations_list)),int(Temperature),int(number_generations),int(number_spins),str(proba).replace('.','_'))
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    filename = folderpath + 'mutations_{}_{}_temperature_{}_generations_{}_number_spins_{}_starteqchain_p{}_seed{}_filenbr{}.h5'.format(int(min(mutations_list)),int(max(mutations_list)),int(Temperature),int(number_generations),int(number_spins),str(proba).replace('.','_'),int(seed),nbrf)
    file = h5py.File(filename, 'w')
    file.close()
    para = [number_spins, Temperature, number_generations,seed,proba]
    file = h5py.File(filename, 'r+')
    file.create_dataset('Parameters', data = np.array(para))
    file.create_dataset('Matrix_contact', data = matcontact)
    file.create_dataset('Chains', data = final_chains, compression='gzip', compression_opts=9)
    file.create_dataset('Mutations', data = np.array(mutations_list))
    if save_chainsandmutations:
        file.create_dataset('allchains', data = all_chains, compression='gzip', compression_opts=9)
        file.create_dataset('mutations', data = mutations_list, compression='gzip', compression_opts=9)
    file.close()

###############################################################################            
##### VARY temperature instead of the number of mutation, uncomment
##### the following section and comment the one above
###############################################################################
# number_averages = 10
# number_mutations = 5
# number_generations = 11
# number_spins = 200
# matcontact =  np.load('./contact_maps/N200p0_02/contactmat_n200p0_02.npy')
# proba = 0.02
# path_to_eq_seq = './example/equilibrium/2022_09_14_12_24_42_Nspins200_probagraph0_02_flips300_Nchains2048_seed_17_filenbr0.h5'
# #change the list of temperature below
# temperaturelist = [1,2,3,4.2,5,6,7]
# for nbrfile in range(0,number_averages):
#     seed = 3*(nbrfile+1)
#     random.seed(seed)
#     final_chains = np.zeros((len(temperaturelist),pow(2,number_generations),number_spins),dtype = np.int8)
    
#     save_chainsandmutations = False
#     if save_chainsandmutations:
#         all_chains = np.zeros((len(temperaturelist),2*pow(2,number_generations)-1,number_spins))  
#         # savedmutations_list = np.zeros((len(mutations_list),2*pow(2,number_generations)-2, number_mutations))
  
#     for idxt, temperature in tqdm(enumerate(temperaturelist)):
#             date = today.strftime("%Y_%m_%d_")
#             hour = startTime.strftime('%H_%M_%S_')
#             Tree = Generate_tree(number_generations)
#             starting_chain = np.zeros(number_spins, dtype = np.int8)
#             # start from equilibrium chain
#             starting_chain = Load_eq_sequence(path_to_eq_seq, 4.0)
#             Tree['0/1'] = starting_chain
#             if save_chainsandmutations:
#                 all_chains[idxt,0,:] = starting_chain
#                 ct = 0
#             for g in range(1,number_generations+1):  
#                 list_parents = np.linspace(1,pow(2,g-1),pow(2,g-1), dtype = np.int16)
#                 for parent in list_parents:
#                     chain = Tree['{}/{}'.format(g-1,parent)]
#                     newchain1,ml1 = Mutate(chain.copy(), number_mutations, temperature, matcontact)
#                     newchain2,ml2 = Mutate(chain.copy(), number_mutations, temperature, matcontact)
     
#                     if save_chainsandmutations:
#                         all_chains[idxt,ct+1,:] = newchain1
#                         all_chains[idxt,ct+2,:] = newchain2
#                         ct = ct+2
#                     Tree['{}/{}'.format(g,2*parent-1)] = newchain1
#                     Tree['{}/{}'.format(g,2*parent)] = newchain2
#             for index_chain, child in enumerate(list(np.linspace(1,pow(2,number_generations),pow(2,number_generations),dtype = np.int16))):
#                 final_chains[idxt,index_chain,:] = Tree['{}/{}'.format(number_generations,child)]
            
        
#     path_store_data = './example/save_folder_example/' 
#     folderpath = path_store_data + 'temperatures_{}_{}_mutation_{}_generations_{}_number_spins_{}_starteqchain_p{}/'.format(int(min(temperaturelist)),int(max(temperaturelist)),int(number_mutations),int(number_generations),int(number_spins),str(proba).replace('.','_'))
#     if not os.path.exists(folderpath):
#         os.makedirs(folderpath)

    
#     filename = folderpath + 'temperatures_{}_{}_mutation_{}_generations_{}_number_spins_{}_starteqchain_p{}_filenbr{}.h5'.format(int(min(temperaturelist)),int(max(temperaturelist)),int(number_mutations),int(number_generations),int(number_spins),str(proba).replace('.','_'),nbrfile)
    
#     file = h5py.File(filename, 'w')
#     file.close()
    
#     para = [number_spins, number_mutations, number_generations,seed,proba,nbrfile]
    
#     file = h5py.File(filename, 'r+')
#     file.create_dataset('Parameters', data = np.array(para))
#     file.create_dataset('Matrix_contact', data = matcontact)
#     file.create_dataset('Chains', data = final_chains, compression='gzip', compression_opts=9)
#     file.create_dataset('Temperatures', data = np.array(temperaturelist))
    
#     if save_chainsandmutations:
#         file.create_dataset('allchains', data = all_chains, compression='gzip', compression_opts=9)
#     file.close()


