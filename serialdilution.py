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

    # ## generate fitness landscape
    # # set up real epistatic vector
    # epistatic_sparsity = 0.05
    # J_real_mask = np.random.binomial(1,epistatic_sparsity,size=(K**L))
    # J_real_sign = 2*np.random.binomial(1,0.5,size=(K**L)) - 1
    # J_real_magnitude = np.random.normal(4,3,size=(K**L))

    # J_real = J_real_magnitude * J_real_sign * J_real_mask

    # # generate real fitness landscape
    # F_real = np.matmul(hadamard(K**L), J_real) / 1000

    # F_real += -np.min(F_real)

    # print(F_real,np.min(F_real),np.max(F_real))

    # J_real = K**(-L) * hadamard(K**L) @ F_real

    ## generate fitness landscape
    # # set up real epistatic vector
    # epistatic_sparsity = 0.05
    # J_real_mask = np.random.binomial(1,epistatic_sparsity,size=(K**L))
    # J_real_magnitude = np.random.normal(0,0.01,size=(K**L))

    # J_real = J_real_magnitude * J_real_mask
    # J_real[0] = 0

    # # generate real fitness landscape
    # F_real = np.matmul(hadamard(K**L), J_real)

    # J_real = K**(-L) * hadamard(K**L) @ F_real

    ## serial dilution model simulation
    starttime = time.time()

    # declare populations
    # populations are initialized to random sequence
    random_start_ind = np.random.choice(K**L)
    # random_start_ind = np.argmin(F_real) # populations are initialized to the lowest fitness
    Gamma_ind = random_start_ind*np.ones((N),dtype=int)

    # keep track of frequency time series
    freq_timeseries = np.zeros((K**L,T))

    # number of time steps per generation
    tau = 4

    # number of offspring individuals
    m = 2

    # replication probabilities
    pi = np.random.beta(2,5,size=(K**L))
    F_real = m*pi/tau

    print(pi)
    print(np.where(pi < 0),np.where(pi > 1))
    print(np.min(pi),np.max(pi))

    # convert Gamma_ind to Gamma_seq
    def Gamma_ind_to_seq(Gamma_ind):
        Q = Gamma_ind.shape[0]
        Gamma_seq = np.zeros((Q,L))
        for q in range(Q):
            Gamma_seq[q,:] = ind2sub(Gamma_ind[q],K,L)
        return Gamma_seq

    # convert Gamma_seq to Gamma_ind
    def Gamma_seq_to_ind(Gamma_seq):
        Q = np.shape(Gamma_seq)[0]
        Gamma_ind = np.zeros((Q))
        for q in range(Q):
            Gamma_ind[q] = sub2ind(Gamma_seq[q,:],K,L)
        return Gamma_ind

    # population size trajectory
    population_size = []

    # run simulation
    for t in tqdm(range(T)):
        Gamma_seq = Gamma_ind_to_seq(Gamma_ind)
        for i in range(tau):
            # record population size
            population_size.append(Gamma_ind.shape[0])

            # determine which individuals are replicating
            replication_probs = np.zeros((Gamma_ind.shape[0]))
            for j in range(Gamma_ind.shape[0]):
                replication_probs[j] = pi[int(Gamma_ind[j])]
            try:
                replication_events = np.random.binomial(1, replication_probs, size=(Gamma_ind.shape[0]))
            except:
                print(replication_probs)

            # get indices of replicators, make m copies
            replicators_ind = Gamma_ind[replication_events == 1]
            replicators_ind = np.array(np.concatenate((replicators_ind, np.tile(replicators_ind,m-1))),dtype=int)
            replicators_seq = Gamma_ind_to_seq(replicators_ind)
            replicators_seq = np.vstack((replicators_seq, np.tile(replicators_seq, (m-1,1))))

            # delete the indices of the original replicators
            Gamma_ind = np.delete(Gamma_ind, np.where(replication_events))
            Gamma_seq = np.delete(Gamma_seq, np.where(replication_events), axis=0)

            # mutate the new replicators
            temp_seq = 2*replicators_seq - 1
            mut_mask = np.random.binomial(1,mu_rate,size=replicators_seq.shape)
            temp_seq *= (-1) * (2*mut_mask - 1)
            mutated_replicators_seq = (temp_seq + 1)/2
            mutated_replicators_ind = Gamma_seq_to_ind(mutated_replicators_seq)

            # update the population 
            Gamma_ind = np.array(np.concatenate((Gamma_ind,mutated_replicators_ind)),dtype=int)
            Gamma_seq = np.vstack((Gamma_seq,mutated_replicators_seq))

        # new broth
        Gamma_ind = np.random.choice(Gamma_ind,N)
            
        # update frequency time series
        unique, counts = np.unique(Gamma_ind, return_counts=True)
        freq_temp = counts / N
        for k in range(len(unique)):
            freq_timeseries[unique[k],int(t)] = freq_temp[k]

    ## make mutation matrix
    path_adj = np.array([[0, 1], [1, 0]])
    M = mu_rate * (make_adj_mat(path_adj, L) - L*(K-1)*np.eye(K**L))

    ## save to file
    file_suffix = '_L' + str(L) + '_N' + str(N) + '_T' + str(T) + '_mu' + str(mu_rate) + '_trial' + str(trial)
    np.savetxt(outdir + '/serialdilution_freq_timeseries' + file_suffix + '.csv', freq_timeseries,delimiter=',')
    np.savetxt(outdir + '/serialdilution_M' + file_suffix + '.csv', M,delimiter=',')
    np.savetxt(outdir + '/serialdilution_F_real' + file_suffix + '.csv', F_real,delimiter=',')
    np.savetxt(outdir + '/serialdilution_population_size' + file_suffix + '.csv', np.array(population_size),delimiter=',')