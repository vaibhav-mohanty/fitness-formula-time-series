import numpy as np
import scipy.stats
import argparse
import pickle
import pandas as pd


# function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Time series frequency data and mutation matrix.')
    parser.add_argument('--model',type=str,help='Model type.')
    parser.add_argument('--L',type=int,help='Sequence length.')
    parser.add_argument('--N',type=int,help='Population size.')
    parser.add_argument('--T',type=int,help='Number of time steps.')
    parser.add_argument('--mu',type=float,help='Mutation rate.')
    parser.add_argument('--trial',type=int,help='Trial number.')
    parser.add_argument('--dir',type=str,help='Output directory.')
    args = parser.parse_args()
    return args

# function to perform fitness estimation from theory
def estimate_fitness(freq_timeseries,M):
    # timeavg_freq = np.sum(freq_timeseries,axis=1)
    # possible_indices = np.where(timeavg_freq > 0)[0]
    # index_to_remove = np.where(timeavg_freq[possible_indices] == np.min(timeavg_freq[possible_indices]))[0]
    # if len(index_to_remove) > 0:
    #     index_to_remove = np.random.choice(index_to_remove,size=1)
    # print(index_to_remove)
    # freq_timeseries = np.delete(freq_timeseries,index_to_remove,axis=0)
    # timeavg_freq = np.delete(timeavg_freq,index_to_remove)

    T = freq_timeseries.shape[1]

    # construct A matrix
    timeavg_freq = np.sum(freq_timeseries,axis=1)
    diag_timeavg_freq = np.diag(timeavg_freq)
    outer_freq = np.einsum('it,jt->ij',freq_timeseries,freq_timeseries)
    A = (diag_timeavg_freq - outer_freq)
    boundary_term = -(freq_timeseries[:,-1] - freq_timeseries[:,0])
    A = np.hstack((A, np.array([boundary_term]).T)) 
    print(np.max(np.linalg.pinv(A)))

    # remove empty rows and columns perform linear regression
    unvisited_genotypes = np.where(timeavg_freq == 0)[0]
    A_reduced = np.delete(A,unvisited_genotypes,axis=0)
    A_reduced = np.delete(A_reduced,unvisited_genotypes,axis=1)
    y = - M @ timeavg_freq
    y_reduced = np.delete(y,unvisited_genotypes)
    F_est = np.linalg.pinv(A_reduced,rcond=1e-6) @ y_reduced
    print(np.max(np.linalg.pinv(A_reduced)))

    print(np.linalg.pinv(A_reduced))

    print(F_est[-1])

    return F_est[:-1], timeavg_freq/T, unvisited_genotypes
    # return F_est, timeavg_freq/T

# function to calculate Pearson r values at various rare frequency cutoffs
def get_correlation_values(F_est,F_real,timeavg_freq,rare_cutoffs):
    pearson_values = np.zeros((len(rare_cutoffs)))
    spearman_values = np.zeros((len(rare_cutoffs)))
    N_common_variants = np.zeros((len(rare_cutoffs)))
    for i in range(len(rare_cutoffs)):
        cutoff = rare_cutoffs[i]
        F_est_common = F_est[timeavg_freq > cutoff]
        F_real_common = F_real[timeavg_freq > cutoff]

        N_common_variants[i] = len(np.where(timeavg_freq > cutoff)[0])
        if len(np.where(timeavg_freq > cutoff)[0]) >= 2:
            pearson = scipy.stats.pearsonr(F_est_common, F_real_common)[0]
            pearson_values[i] = pearson

            spearman = scipy.stats.spearmanr(F_est_common, F_real_common)[0]
            spearman_values[i] = spearman
        else:
            pearson_values[i] = np.nan
            spearman_values[i] = np.nan

    correlation_data = pd.DataFrame({'rare_cutoffs_log10': np.log(rare_cutoffs)/np.log(10), 'N_common_variants': N_common_variants, 'pearson_values': pearson_values, 'spearman_values': spearman_values})
    print(correlation_data)
    return correlation_data

# main
if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    modeltype = args.model
    L = args.L
    N = args.N
    T = args.T
    mu_rate = args.mu
    trial = args.trial
    filedir = args.dir

    filesuffix = '_L' + str(L) + '_N' + str(N) + '_T' + str(T) + '_mu' + str(mu_rate) + '_trial' + str(trial)
    datapath = filedir + '/' + modeltype + '_freq_timeseries' + filesuffix + '.csv'
    mutationpath = filedir + '/' + modeltype + '_M' + filesuffix + '.csv'
    fitnesspath = filedir + '/' + modeltype + '_F_real' + filesuffix + '.csv'
    outpath = filedir + '/' + modeltype + '_processed' + filesuffix + '.pkl'

    # get time series data, data assumed to have shape [Nvariants, T]
    freq_timeseries = np.loadtxt(datapath,delimiter=',')

    timeavg_freq = np.sum(freq_timeseries,axis=1)
    possible_indices = np.where(timeavg_freq > 0)[0]
    index_to_remove = np.where(timeavg_freq[possible_indices] == np.min(timeavg_freq[possible_indices]))[0]
    if len(index_to_remove) > 0:
        index_to_remove = np.random.choice(index_to_remove,size=1)
    print(index_to_remove)
    freq_timeseries = np.delete(freq_timeseries,index_to_remove,axis=0)

    T = freq_timeseries.shape[1]
    Nvariants = freq_timeseries.shape[0]

    # get mutation matrix
    M = np.loadtxt(mutationpath,delimiter=',')

    # get real fitnesses
    F_real = np.loadtxt(fitnesspath,delimiter=',')

    # delete lowest frequency fitness to improve matrix conditioning
    M = np.delete(M,index_to_remove,axis=0)
    M = np.delete(M,index_to_remove,axis=1)
    F_real = np.delete(F_real,index_to_remove)
    Nvariants -= 1

    # get estimated fitnesses, timeavg_freq
    F_est_reduced, timeavg_freq, unvisited_genotypes = estimate_fitness(freq_timeseries,M)
    F_real_reduced = np.delete(F_real, unvisited_genotypes)
    timeavg_freq_reduced = np.delete(timeavg_freq, unvisited_genotypes)
    F_est = np.zeros((len(F_real)))
    F_est[np.delete(np.arange(len(F_real)),unvisited_genotypes)] = F_est_reduced
    F_est[unvisited_genotypes] = np.nan

    # get Pearson r at various cutoffs
    rare_cutoffs = 10 ** np.arange(-5,-0,0.25)
    correlation_data = get_correlation_values(F_est_reduced,F_real_reduced,timeavg_freq_reduced,rare_cutoffs)

    # save outputs to file
    data = {'F_real': F_real,
            'F_real_reduced': F_real_reduced,
            'F_est': F_est,
            'F_est_reduced': F_est_reduced,
            'timeavg_freq': timeavg_freq,
            'timeavg_freq_reduced': timeavg_freq_reduced,
            'unvisited_genotypes': unvisited_genotypes,
            'rare_cutoffs': rare_cutoffs,
            'correlation_data': correlation_data}
    
    with open(outpath,'wb') as file:
        pickle.dump(data,file)