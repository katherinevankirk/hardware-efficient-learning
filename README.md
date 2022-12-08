# hardware-efficient-learning
Hardware-efficient learning of quantum many-body states. Code for simulating a U(1) lattice gauge theory and classifying topological order.

# Hardware-efficient learning of quantum many-body states.
This is an open source implementation for the paper "Hardware-efficient learning of quantum many-body states."

We require g++ and python version 3.  

### Quick Start
Do the following in the terminal.


```
# 
# Python code "LGTNumerics.ipynb" is used for estimating the energy density of a lattice gauge theory, 
# and all remaining code is used for classifying topological phases.
#

## APPLICATION I. ESTIMATING ENERGY DENSITY OF LATTICE GAUGE THEORY
## Open the LGT iPython notebook: LGTNumerics.ipynb.
> jupyter notebook


## APPLICATION II. CLASSIFYING TOPOLOGICAL PHASES
## The data for this application was generated and manipulated on the Harvard cluster. Therefore, the 
## numerics were performed in batches. The first three steps were data generation/manipulation (slurm
## files call the cpp and python code). For the last step, open the phases iPython notebook: 
## ClassifyPhases.ipynb. Finally, while the code below is written for a cluster, this is not required. 
## One can simply run the commands in the slurm files directly on the terminal and modify the .py
## codes to require fewer batches.

> sbatch slurm_createtoptrivstates.sh    # Step 1. Create topological & trivial states.
> sbatch slurm_globalsu2.sh              # Step 2. Perform global su2 shadow tomography.
> sbatch slurm_kerneldata.sh             # Step 3. Collect all global shadow data into shadow kernel.
> jupyter notebook                       # Step 4. Kernel PCA to predict phases.

```
