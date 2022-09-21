# Phylogeny-contact-inference
## Getting started

Clone this repository on your local machine by running
```bash
git clone git@github.com:Bitbol-Lab/Phylogeny-contact-inference.git
```
and move inside the root folder.

Then, install the required libraries:
```python
pip install -r requirements.txt
```
### 2 states
The 2states folder contains two scripts to generate sequences. First, ```Generate_sequences_equilibrium_binaryvariables.py``` allows to generate independent sequences at equilibrium. Second, ```Generate_sequences_phylogeny_binaryvariables.py``` allows to generate sequences with phylogeny. Note that an equilibrium dataset is needed to generate sequences with phylogeny because an equilibrium sequence is taken as the root of the phylogeny. 
Then, there is a script, ```Inference_binaryvariables.py```, for the inference task that implements the Mutual Information, Covariance and mean-field DCA methods. For plmDCA, we used the ```PlmDCA``` package (https://github.com/pagnani/PlmDCA) and saved the coupling matrix in the ```.jld``` format which is then taken as input by the script.
Finally, the script ```Compute_characteristics_G_N_L.py``` computes the characteristics of pairs of sites (G, L, N). 
Sequences are included in the example folder, providing a working example for the inference and the computation of the characteristics (G, L, N).

### 21 states
This folder contains scripts to compute the phylogenetic and network characteristics (G', L, N), ```Analyse_FP_realistic_sequences.py```, and the PPV of contact prediction from coevolution score matrices, ```ComputePPV_realisticseq.py```. Scripts run with provided examples of different datasets along with scores computed using Mutual Information and plmDCA. The plmDCA inference and the phylogenetic reweighting are performed using the ```PlmDCA``` package.
