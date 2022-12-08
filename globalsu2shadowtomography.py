#Katherine Van Kirk
#kvankirk@g.harvard.edu

# PACKAGES
# --------

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import copy
import math
from math import e
import random 
import ast
import time
import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import svds, eigs
from itertools import permutations
from scipy.stats import unitary_group
from numpy.random import choice
from scipy.linalg import null_space, expm
from scipy.linalg import norm
from scipy.stats import unitary_group
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import jax
from neural_tangents import stax
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import copy
import sys

# A. Helper Functions for Creating Efficient Representations of 'Bi's
# -------------------------------------------------------------------
# 'Efficient Representation' definition: For a Bi, take all the pauli strings in the Bi's sum and put them in 
# an array. For example, represent XY + YX by ['XY','YX']

# DESCRIPTION: Insert 'elem' in the string at site i
def str_insert(i,elem,string):
    return string[:i] + elem + string[i:]

# DESCRIPTION: Returns array of permutations of 'string.' When any of n_x, n_y, x_z, or n_I is greater than 1, 
# just permuting the string will yield duplicates-- the below removes duplicates. 
def string_permute(string):
    return list(set(["".join(perm) for perm in permutations(string)]))

# DESCRIPTION: From a string of X,Y,Z,I paulis, create the corresponding B_i element. This element in the form 
# of an array-- each element in the array is a different pauli string A_k within the sum that makes up B_i. 
def string_permute_non_is(string):
    #permutes all non_Is by removing Is
    np_str = np.array(list(string)) 
    I_mask = np_str == "I"
    other = np_str[~I_mask] 
    perms = string_permute("".join(other)) 
    
    #place Is back in and return
    new = []
    for perm in perms:      
        new_elem = ""
        j = 0
        for is_I in I_mask:
            if is_I:
                new_elem += "I"  
            else: 
                new_elem += perm[j]
                j += 1
        new.append(new_elem)
    return new

# DESCRIPTION: Given a set of Bis for some system size n, but the set has duplicates. Here we remove the (unwanted)
# duplicates.
def de_duplicate(B):
    arr = [",".join(sorted(x)) for x in B]
    arr = list(set(arr))
    return [elem.split(",") for elem in sorted(arr)]

# DESCRIPTION: Counts size of visible space for a n-qubit system
def visible_space_size(n):
    return int((2**(n-3)) * (n**2 + 7*n + 8))

# DESCRIPTION: Given a n-1 length string b, add an identity to every location. Of course, for bs like XXII this 
# creates redundancy. However, this is ok-- we will remove redundancy later. 
def add_identity(b,n):
    new = []
    for i in range(n):
        x = copy.deepcopy(b)
        b2 = [str_insert(i,"I",x_comp) for x_comp in x]
        new.append(b2)
    return new

# DESCRIPTION: Given a n-1 length string b, add matrix (X, Y or Z) to every location. Since this is non-I, we must
# also permute the XYZ elements after adding a new one to the set. Then we return the corresponding set of new, 
# length-n bis. 
def add_non_identity(b,matrix,n):
    new = []
    for i in range(n):
        x_first = copy.deepcopy(b[0])
        new_first = str_insert(i,matrix,x_first)
        new.append(string_permute_non_is(new_first))
    return new

# DESCRIPTION: Creates the visible basis in the efficient, string format defined at the start of this section. The 
# only input is the system size "n"
def visible_basis_strings(n):
    # a. Create the sets of pauli strings corresponding to each Bi. Build up to Bis for system size n by using Bis for 
    #    smaller system sizes. 
    B = {1: ["I","X","Y","Z"]} 
    for sz in range(2,n+1):
        Bsz = []
        for bi in B[sz-1]:
            for matrix in ["I","X","Y","Z"]:
                if matrix == "I":
                    #add it to every spot
                    Bsz.extend(add_identity(bi,2))
                else:
                    Bsz.extend(add_non_identity(bi,matrix,2))
        B[sz] = de_duplicate(Bsz)
    return B[n]


# B. Helper Functions for Evaluating our 'Bi' Quantity of Interest
# ----------------------------------------------------------------

# DESCRIPTION: Given a specific pauli string in some bi. Using this string (in the form of a string object), this
# function constructs the corresponding matrix. 
def pauli_string_to_matrix(string):
    # Make a vector of the pauli matrices
    chars = list(string)
    flow = []
    for i in chars:
        if i == "I":
            flow.append(I)
        if i == "X":
            flow.append(X)
        if i == "Y":
            flow.append(Y)
        if i == "Z":
            flow.append(Z)
    res = flow[0]
    
    # Tensor them together
    for j in range(1,len(flow)):
        res = np.kron(res,flow[j])
        
    return res

# DESCRIPTION: Given a specific bi in the form for example bi = [IIXXY, IIXYX, IIYXX]. For each internal string
# in the bi sum, you construct the matrix (using pauli_string_to_matrix) and then add them together. 
def bi_to_matrix(bi):
    res = pauli_string_to_matrix(bi[0])
    n = len(bi[0])
    for i in range(1,len(bi)):
        res += pauli_string_to_matrix(bi[i])
    
    return math.sqrt(1/(2**n * len(bi))) * res

# DESCRIPTION: Return the visible basis for a system size n. The format of the returned data structure is a list
# of sparse csr matricies. 
# B is dictionary of all possible bi's for a set of system sizes n=1,...,9 (n is the key)
def visible_basis(n):
    # a. Create the sets of pauli strings corresponding to each Bi. Build up to Bis for system size n by using Bis for 
    #    smaller system sizes. 
    B = {1: ["I","X","Y","Z"]} 
    for sz in range(2,n+1):
        Bsz = []
        for bi in B[sz-1]:
            for matrix in ["I","X","Y","Z"]:
                if matrix == "I":
                    #add it to every spot
                    Bsz.extend(add_identity(bi,2))
                else:
                    Bsz.extend(add_non_identity(bi,matrix,2))
        B[sz] = de_duplicate(Bsz)
    
    # b. Generate the actual matrices for n and return.  
    vis = [csr_matrix(bi_to_matrix(bi)) for bi in B[n]]
    return vis

# DESCRIPTION: Given a density matrix rho and the systemsize, we calculate the visible space expectation values.
# Note that the visible basis is in CSR sparse format, and rho should be passed in as such too. 
def visible_expvals(syssize, rho, vbasis):
    #print([x.todense() for x in vbasis])
    return [np.real(np.trace((rho*bi).todense())) for bi in vbasis]


# C. Implementing M Channel on Blocked Subspaces
# ----------------------------------------------
# Condition for being in the same blocked subspace:
#    1. all I must be on same sites
#    2. n_i + n_j = even set
#       --> (n_{i,x} + n_{j,x}, n_{i,y} + n_{j,y}, n_{i,z} + n_{j,z}) = (even number, even number, even number)

# DESCRIPTION: Given one of the terms within a Bi operator (e.g. given XYI for Bi = XYI+YXI), this function
# returns a list of the indices where there is an identity. 
# example: find_identity('XYII') = [2,3]
def find_identity(bi):
    return [i for i, ltr in enumerate(bi) if ltr == 'I']

# DESCRIPTION: Given two terms within two operators B1 and B2 (e.g. a term for Bi = XYI+YXI could be XYI), function 
# returns whether or not the corresponding B1 and B2 have I on the same sites. 
def identity_same_sites(b1, b2):
    return find_identity(b1) == find_identity(b2)

# DESCRIPTION: Given a term within Bi (e.g. XYI for Bi = XYI+YXI), return tuple of parameters counting the 
# occurences of various paulis: (nI, nX, nY, nZ)
def nvector(bi):
    nI = len([i for i, ltr in enumerate(bi) if ltr == 'I'])
    nX = len([i for i, ltr in enumerate(bi) if ltr == 'X'])
    nY = len([i for i, ltr in enumerate(bi) if ltr == 'Y'])
    nZ = len([i for i, ltr in enumerate(bi) if ltr == 'Z'])
    return (nI, nX, nY, nZ)

# DESCRIPTION: Given two terms within two operators B1 and B2 (e.g. a term for Bi = XYI+YXI could be XYI), 
# function checks whether the n-tuples corresponding to B1 and B2 sum to an all-even tuple. 
def even_combined_components(b1, b2):
    (n1I, n1X, n1Y, n1Z) = nvector(b1)
    (n2I, n2X, n2Y, n2Z) = nvector(b2)
    if (n1X + n2X)%2 == 1: return False 
    if (n1Y + n2Y)%2 == 1: return False
    if (n1Z + n2Z)%2 == 1: return False
    
    return True

# DESCRIPTION: Given two terms within two operators B1 and B2 (e.g. a term for Bi = XYI+YXI could be XYI), 
# function checks whether the corresponding B1 and B2 live within the same block. 
def in_same_block(b1, b2):
    return identity_same_sites(b1, b2) and even_combined_components(b1, b2)

# DESCRIPTION: Given system size n, return a list of blocks. The structure of this list will be as follows:
#   - each each element in the block list will be a list of the Bis living in the block
#   - each Bi is represented by a list of the pauli strings that make it up
def create_blocks(n):
    #A. Make list of blocks
    blockdict = []
    
    for Bi in visible_basis_strings(n):
        b1 = Bi[0]
        appended = False
        
        # i. check if current Bi lives in any of the existing blocks
        for x in range(len(blockdict)):
            block = blockdict[x]
            b2 = block[0][0]
            if in_same_block(b1,b2): 
                block.append(Bi)
                blockdict[x] = block
                appended = True
        
        # ii. if current Bi does not live in any existing block, add to new block
        if appended == False:
            blockdict.append([Bi])
    
    #B. Return list of blocks
    return blockdict

# DESCRIPTION: Given two terms within two operators B1 and B2 (e.g. a term for Bi = XYI+YXI could be XYI), 
# calculate the c super-matrix element between the two Bi operators. 
def cmatrix(b1,b2):
    (n1I, n1X, n1Y, n1Z) = nvector(b1)
    (n2I, n2X, n2Y, n2Z) = nvector(b2)
    k = int((n1X+n1Y+n1Z+n2X+n2Y+n2Z)/2)
    cnumerator = 2*math.factorial(k)*math.factorial(k+1)*math.factorial(n1X+n2X)*math.factorial(n1Y+n2Y)*math.factorial(n1Z+n2Z) 
    cdenominator = math.factorial(2*k+2) * \
                   math.factorial((n1X+n2X)//2) * \
                   math.factorial((n1Y+n2Y)//2) * \
                   math.factorial((n1Z+n2Z)//2) * \
                   math.sqrt(math.factorial(n1X) * \
                   math.factorial(n2X)*math.factorial(n1Y) * \
                   math.factorial(n2Y)*math.factorial(n1Z) * \
                   math.factorial(n2Z))
    return cnumerator/cdenominator

# DESCRIPTION: Given a block of the M superoperator, return a list of the eigenvalues on that block
# note that all evals are real and positive
def evals_of_block(block):
    #make the c matrix for the block
    cmat = np.zeros((len(block),len(block)))
    for r in range(len(block)):
        for c in range(len(block)):
            cmat[r][c] = cmatrix(block[r][0],block[c][0])
    #diagonalize to get evals
    return np.real(linalg.eigvals(cmat))

# DESCRIPTION: Given a block of the M superoperator, return the eigendecompositon on that block in terms of
# a tuple: (eigenvalues, eigenvectors)
# note that all evals are real and positive
def eigdecomp_of_block(block):
    #make the c matrix for the block
    cmat = np.zeros((len(block),len(block)))
    for r in range(len(block)):
        for c in range(len(block)):
            cmat[r][c] = cmatrix(block[r][0],block[c][0])
    #diagonalize to get evals
    vals, vecs = linalg.eigh(cmat) # EIG DOES NOT RETURN ORTHONORMAL VECS UGHHHHH.e
    return (np.real(vals),vecs)

# DESCRIPTION: Given system size n, return a list of all eigenvalues of the M superoperator.
def evals_of_visible(n):
    blocks = create_blocks(n)
    evals = []
    for block in blocks:
        evals.extend(evals_of_block(block)) # add evals from each block
    return np.sort(evals)

# DESCRIPTION: Given system size n, you have some set of eigenvalues. Chop off the smallest percent p (e.g. p = 0.1)
# corresponds to chopping off smallest 10%. Then return the smallest eigenvalues of the NEW set. 
def chop_smallest_evals(n, p):
    allevals = evals_of_visible(n) #these are returned sorted smallest-->largest
    sizeevals = visible_space_size(n)
    indexnewsmallest = int(np.floor(sizeevals*p))
    return allevals[indexnewsmallest]

# DESCRIPTION: Calculate the variance of each eigenvector of the M channel (i.e. 1/eval for the corresponding)
# eigenvector. 
def average_variance_visible(n):
    evals = evals_of_visible(n)
    var = [1/ev for ev in evals]
    return sum(var)/len(var)

# DESCRIPTION: Given a list of Bis and the corresponding weights of those Bis (i.e. the values in vec), contruct
# the eigenvector operators of the M channel.
# block = [B1, B2, ...]
# vec = [amount of B1, amount of B2, ...]
def make_evec(block, vec, n):
    evecOp = np.zeros((2**n,2**n))
    for x in range(len(block)):
        bi = bi_to_matrix(block[x])
        evecOp = np.add(evecOp, np.multiply(vec[x],bi))
    return evecOp

# DESCRIPTION: Gives the eigenvalus and eigenvectors of the M channel. This serves as setting up the M^-1 channel 
# because we can invert via multiplying evec components by 1/eval
# NOTE: output looks like... [(eval1, evec1), (eval2, evec2), ... ]
def setup_MChannel_EvalEvecs(n):
    blocks = create_blocks(n)
    mChann = []
    for block in blocks:
        vals, vecs = eigdecomp_of_block(block)
        vecs = np.transpose(vecs) #new
        #print('\n',block)
        #print('eigenvalues: ', vals)
        #print('eigenvectors: ', vecs)
        for x in range(len(vals)):
            mChann.append((vals[x], np.matrix(make_evec(block, vecs[x],n)))) #tuple of val/vec
    return mChann

# DESCRIPTION: Given a state \rho (de-densified), returns M^{-1}(\rho). You should pass in the (eval, evec)
# form of the M channel, which was generated with the 'setup_MChannel_EvalEvecs' function.
def implement_inverseMChannel(rho, mChann, n, val_threshold = 0):
    vals = np.array([entry[0] for entry in mChann if entry[0] > val_threshold])
    vecs = np.stack([np.array(entry[1]) for entry in mChann if entry[0] > val_threshold]) 
    rho = np.array(rho).reshape(1,2**n,2**n)
    coeffs = np.trace(rho @ vecs, axis1=1,axis2=2)/vals
    coeffs = coeffs.reshape(-1,1,1)
    
    rhoprime = np.sum(coeffs * vecs,axis = 0)
    return rhoprime

# DESCRIPTION: Given a state \rho (de-densified), returns M(\rho). You should pass in the (eval, evec)
# form of the M channel, which was generated with the 'setup_MChannel_EvalEvecs' function.
def implement_MChannel(rho, mChann, n):
    rhoprime = np.zeros((2**n,2**n))
    for (val, vec) in mChann:
        if val > 0.1: #NEW 11/9: for n=5, threshhold cuts off 25% smallest evals
            coeff = np.trace(np.matrix(rho)*np.matrix(vec))
            coeff = coeff*val
            rhoprime = np.add(rhoprime, np.multiply(coeff, vec))
    return rhoprime


# D. Perform globalsu2 shadow tomography
# --------------------------------------


# DESCRIPTION: Generates a random su2 unitary and tensors it n times for an n-qubit system
def random_globalsu2(n):
    V = unitary_group.rvs(2) #U(2)
    Vtensorn = V
    for x in range(n-1):  # Kronecker n times
        Vtensorn = np.kron(Vtensorn, V)
    return Vtensorn

# DESCRIPTION: this function returns the chosen computational basis element found via measurement (It returns the
# chosen vector in the form of the computational basis element's index). 
# GIVEN-- rho in the form of a array of rho's diagonal elements 
def measurement(n, rho):
    #bprobs = np.real(rho) #original
    positiverho = np.array(rho)
    positiverho[positiverho < 0] = 0
    bprobs = np.abs(positiverho)/np.sum(np.abs(positiverho)) #the classical shadow from random pauli data won't be perfect
    bvecs = range(2**n)
    draw = choice(bvecs, 1, p = bprobs)
    return draw[0]

# DESCRIPTION: returns non-sparse density matrix for the b><b state post mmt
# mmt: index of the computational basis state that was measured
def makebbvector(mmt, n):
    init_state = np.zeros((2**n,2**n))
    init_state[mmt][mmt] = 1
    return init_state

# DESCRIPTION: Constructs a single shadow of the n-qubit state 'state'
def make_globalsu2_shadow(n, state, mChann):
    # 1. Make the measurement 
    Vtensorn = csc_matrix(random_globalsu2(n))
    vstatev = Vtensorn * state * Vtensorn.getH() 
    mmt = measurement(n, np.diag(vstatev.todense())) #have to pass in only diagonal components of the state
    # 2. Make corresponding classical shadow
    shadow = csc_matrix(makebbvector(mmt, n)) #make b><b vector
    shadow = Vtensorn.getH() * shadow * Vtensorn #vtensorn
    shadow = implement_inverseMChannel(shadow.todense(), mChann, n) #M inverse
    return shadow
    
# DESCRIPTION: Implements the shadow tomography procedure on the state and returns the average classical shadow
# after making 'Nmmts' shadows. 
#    - n: number of qubits in subsystem
#    - Nmmts: number of shadows
def shadow_tomography(n, state, Nmmts, mChann):
    collectShadows = np.zeros((2**n,2**n))
    for m in range(Nmmts):#Make the shadow measurements
        collectShadows = np.add(collectShadows, make_globalsu2_shadow(n, csc_matrix(state), mChann))
    return np.multiply(collectShadows, 1/Nmmts)

# DESCRIPTION: Implements shadow tomography procedure and uses the resultant set of classical shadows to make 
# expectation value estimates for all the visible space basis elements. 
def shadow_estimated_expvals(n, state, Nmmts, mChann):
    stateestimate = shadow_tomography(n, state, Nmmts, mChann)
    vbasis = visible_basis(n)
    return [np.real(np.trace(bi.dot(stateestimate))) for bi in vbasis]


# E. Translate XYZ measurements to perfect shadows
# ------------------------------------------------

# DESCRIPTION: This function sets up the shadows associated with each of the possible Pauli measurement (X,Y,Z) 
# outcomes on a single qubit. 
def makePauliShadows():
    singleshadow = []
    bzero = [[1,0],[0,0]]
    bone = [[0,0],[0,1]]
    
    #Z measurement
    singleshadow.append(bzero)
    singleshadow.append(bone)
    
    #X measurement
    H = [[np.cos(math.pi/4),np.sin(math.pi/4)],[-np.sin(math.pi/4),np.cos(math.pi/4)]] #Hadamard
    singleshadow.append(np.matmul(np.matmul(np.transpose(H),bzero), H))
    singleshadow.append(np.matmul(np.matmul(np.transpose(H),bone), H))
    
    #Y measurement
    Phase = [[e ** (0+math.pi*0.25j),0],[0,e ** (0-math.pi*0.25j)]] #90deg rotation CCW
    HP = np.matmul(H,Phase)
    singleshadow.append(np.matmul(np.matmul(np.conjugate(np.transpose(HP)),bzero), HP))
    singleshadow.append(np.matmul(np.matmul(np.conjugate(np.transpose(HP)),bone), HP))
    
    #Apply Depolarizing Channel
    for s in range(len(singleshadow)):
        singleshadow[s] = np.multiply(3,singleshadow[s]) - np.identity(2)
        
    return singleshadow

# DESCRIPTION: This function takes in the data vector and the patch index and returns the corresponding 6-site
# classical shadow. 
def getShadow(datavec, patchIndex):
    indices = [patchIndex, patchIndex+L, patchIndex+(2*L)]
    shadow = 0
    for i in indices: 
        measurementresult = datavec[i]
        if i == patchIndex: 
            shadow = pauliShadows[measurementresult]
        else:
            shadow = np.kron(shadow, pauliShadows[measurementresult])  
    return shadow

# DESCRIPTION: This function takes in a single measurement data vector. It iterates through the patches, creating a 
# (single measurement) classical shadow for each patch and returns an array of these shadows. 
# --> return a vector of shadows where each shadow corresponds to a different patch
def dataVecToShadowPatches(datavec):
    # PBC so edges are fine
    shadowPatches = [getShadow(datavec, pat) for pat in range(PATCHES)]
    return shadowPatches

# DESCRIPTION: This function takes in an array containing many datavectors of a single state. This function adds
# up the shadows made for each datavector (datavector = data from single measurement), where the shadows are added
# up patch-wise. 
def makeNmmtShadowPatchesPerState(state):
    shadowsOfState = []
    for ind, datavec in enumerate(state):
        if ind == 0:
            shadowsOfState = dataVecToShadowPatches(np.transpose(datavec))
        else:
            shadowsOfState = np.add(shadowsOfState, dataVecToShadowPatches(np.transpose(datavec)))
    return np.multiply(1/len(state), shadowsOfState)




# GLOBAL VARIABLES
# ----------------

# Pauli Matrices
I = np.array([[1,0],[0,1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]],dtype=complex)
Z = np.array([[1,0],[0,-1]])

# Constants
L = 10 #side length of toric code
PATCHES = (2*L**2)-(2*L)
DEPTH = 5
NSTATES = 100
GSU2MMTS = 1000
MMTCHANNEL = setup_MChannel_EvalEvecs(3)

statestoiterate = 1 #10
pauliShadows = makePauliShadows() #vector of shadows created from the possible single qubit measurements made


def main(jobnumber):
    
    dep = int(jobnumber/NSTATES)
    stateind = jobnumber%NSTATES

    with open('data{}.txt'.format(jobnumber), 'r') as f:
        cppdata = ast.literal_eval(f.read())
    
    allshadowsarray = makeNmmtShadowPatchesPerState(cppdata)
    print('near perfect shadows are done.')
    
    print('\nworking on patch...')
    stateexpvals = []
    for p in range(PATCHES):
        print(p)
        stateexpvals.append(shadow_estimated_expvals(3, allshadowsarray[p], GSU2MMTS, MMTCHANNEL))
    
    with open('saveallexpvals_threesite_{}_{}.npy'.format(dep,stateind), 'wb') as f:
        np.save(f, np.array(stateexpvals))



if __name__ == "__main__":
    start = time.time()
    
    jobnumber = int(sys.argv[1])
    GSU2MMTS = int(sys.argv[2])
    main(jobnumber)
    
    elapsed = time.time() - start
    print(f"Elapsed time {elapsed/60:.2f} min")



# DESCRIPTION... This code takes in the data generated by torictrivial_datageneration.cpp and creates a set of 
# reduced density matrices on patches of the states -- saved in an np.array indexed by depth, state, and then patch.
    # ...per depth, per state, per patch...
        # a reduced density matrix of the patch
# Then, we perform globalsu2 shadow tomography on each patch, and using the resulting shadow, 
    
# How to read in the final np array: 
# ----------------------------------
# with open('saveallexpvals_threesite_{}_{}.npy'.format(dep,stateind), 'rb') as f:
#     allexpvals = np.load(f)
