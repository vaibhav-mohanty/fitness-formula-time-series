import numpy as np
import math
import time
from tqdm import tqdm
from scipy.linalg import hadamard
import argparse

# converts an index to a Hamming graph position
def ind2sub(ind,K,L):
    if ind == 0:
        return np.zeros(L, dtype=int)
    else:
        mysub = int(np.base_repr(ind,K))
        digits = int(math.log10(mysub))+1
        mysub_str = '0'*(L-digits) + str(mysub)
        unjoined = list(mysub_str)
        return np.array([int(i) for i in unjoined])

# converts a Hamming graph position to an index
def sub2ind(sub,K,L):
    return int(sum(np.multiply(sub,[K**(L-i-1) for i in range(L)])))


# function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Time series frequency data and mutation matrix.')
    parser.add_argument('--L',type=int,help='Sequence length.')
    parser.add_argument('--N',type=int,help='Population size.')
    parser.add_argument('--T',type=int,help='Number of time steps.')
    parser.add_argument('--mu',type=float,help='Mutation rate.')
    parser.add_argument('--trial',type=int,help='Trial number.')
    parser.add_argument('--dir',type=str,help='Output directory.')
    args = parser.parse_args()
    return args

# recursive function to generate adjacency matrix for hypercube
def make_adj_mat(complete_adj,L):
    temp = complete_adj
    if L > 1:
        n = temp.shape[0]
        temp = np.concatenate((np.concatenate((temp, np.eye(n)), axis=1), np.concatenate((np.eye(n), temp), axis=1)), axis=0)
        return make_adj_mat(temp, L-1)
    else:
        return temp

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    K = 2
    L = args.L
    N = args.N
    T = args.T
    mu_rate = args.mu
    trial = args.trial
    outdir = args.dir

    ## generate fitness landscape
    # set up real epistatic vector
    epistatic_sparsity = 0.05
    J_real_mask = np.random.binomial(1,epistatic_sparsity,size=(K**L))
    J_real_magnitude = np.random.normal(0,0.01,size=(K**L))

    J_real = J_real_magnitude * J_real_mask
    J_real[0] = 0

    # generate real fitness landscape
    F_real = np.matmul(hadamard(K**L), J_real)

    J_real = K**(-L) * hadamard(K**L) @ F_real

    ## Moran dynamics simulation
    starttime = time.time()

    # declare populations
    # populations are initialized to random sequence
    random_start_ind = np.random.choice(K**L)
    # random_start_ind = np.argmin(F_real) # populations are initialized to the lowest fitness
    Gamma_ind = random_start_ind*np.ones((N),dtype=int)

    # keep track of frequency time series
    freq_timeseries = np.zeros((K**L,T))

    # run simulation
    for t in tqdm(range(T*N)):
        # choose 2 indices
        candidates = np.random.choice(N, size=2, replace=False) # supposed candidates[0] has been chosen to replace candidates[1]
        
        # find which one has higher fitness
        if np.random.binomial(1,np.exp(F_real[Gamma_ind[candidates[0]]]) / (np.exp(F_real[Gamma_ind[candidates[0]]]) + np.exp(F_real[Gamma_ind[candidates[1]]]))):
            keep_index = candidates[0]
            remove_index = candidates[1]
        else:
            keep_index = candidates[1]
            remove_index = candidates[0]
        
        # get mutated version of candidate to be kept
        temp_seq = ind2sub(Gamma_ind[keep_index],K,L)
        temp_seq = 2*temp_seq - 1 # switches to {+1, 1} basis
        mut_mask = np.random.binomial(1,mu_rate,size=(L))
        temp_seq *= (-1) * (2*mut_mask - 1)
        temp_seq = (temp_seq + 1)/2
        temp_ind = sub2ind(temp_seq,K,L)

        # replace candidate with the mutated candidate
        Gamma_ind[remove_index] = temp_ind
        # Gamma_ind[keep_index] = temp_ind
        
        # update frequency time series
        if t % N == 0:
            unique, counts = np.unique(Gamma_ind, return_counts=True)
            freq_temp = counts / N
            for k in range(len(unique)):
                freq_timeseries[unique[k],int(t/N)] = freq_temp[k]

    ## make mutation matrix
    path_adj = np.array([[0, 1], [1, 0]])
    M = mu_rate * (make_adj_mat(path_adj, L) - L*(K-1)*np.eye(K**L))

    ## save to file
    file_suffix = '_L' + str(L) + '_N' + str(N) + '_T' + str(T) + '_mu' + str(mu_rate) + '_trial' + str(trial)
    np.savetxt(outdir + '/moran_freq_timeseries' + file_suffix + '.csv', freq_timeseries,delimiter=',')
    np.savetxt(outdir + '/moran_M' + file_suffix + '.csv', M,delimiter=',')
    np.savetxt(outdir + '/moran_F_real' + file_suffix + '.csv', F_real,delimiter=',')