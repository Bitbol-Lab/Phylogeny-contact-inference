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
The 2states folder contains two scripts to generate sequences (one at equilibrium and one with phylogeny). Note that an equilibrium dataset of sequences is needed to generate sequences with phylogeny (for the root of the phylogeny). Then, there is a script for the inference task that implements the Mutual Information, Covariance and mean-field DCA methods. For plmDCA, we used the ```PlmDCA``` package and saved the coupling matrix in the ```.jld``` format.
Finally, there is a script to compute the characteristics of pairs of sites (for instance false positives pairs of sites). There should be a working example that runs for the inference and the computation of the characteristics (G, L, N). 

### 21 states
This folder contains the scripts to compute the phylogenetic and network characteristics (G', L, N) and also the PPV. The scripts run with provided examples of different datasets along with scores computed using Mutual Information and plmDCA. The plmDCA inference is also performed using the ```PlmDCA``` package, as well as for the phylogenetic reweighting scheme.
