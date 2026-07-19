import numpy as np
import scipy.stats
import argparse
import pickle
import pandas as pd
import scipy.linalg


# function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Time series frequency data and mutation matrix.')
    parser.add_argument('--model',type=str,help='Model type.')
    parser.add_argument('--nlogrcond',type=float,help='Negative of the log10 of rcond.')
    parser.add_argument('--dir',type=str,help='Output directory.')
    args = parser.parse_args()
    return args

# function to perform fitness estimation from theory
def estimate_fitness(freq_timeseries,M,nlogrcond):
    T = freq_timeseries.shape[1]
    my_rcond = 10**(-nlogrcond)

    # construct A matrix
    timeavg_freq = np.sum(freq_timeseries,axis=1)
    diag_timeavg_freq = np.diag(timeavg_freq)
    outer_freq = np.einsum('it,jt->ij',freq_timeseries,freq_timeseries)
    A = (diag_timeavg_freq - outer_freq)
    boundary_term = -(freq_timeseries[:,-1] - freq_timeseries[:,0]) 
    A = np.hstack((A, np.array([boundary_term]).T)) 
    print(np.max(np.linalg.pinv(A)))
    # print(np.linalg.svd(A))

    # remove empty rows and columns perform linear regression
    unvisited_genotypes = np.where(timeavg_freq == 0)[0]
    A_reduced = np.delete(A,unvisited_genotypes,axis=0)
    A_reduced = np.delete(A_reduced,unvisited_genotypes,axis=1)
    y = - M @ timeavg_freq
    # U,D,V = np.linalg.svd(A_reduced)
    # np.savetxt('A.csv',A)
    y_reduced = np.delete(y,unvisited_genotypes)
    F_est = np.linalg.pinv(A_reduced,rcond=my_rcond) @ y_reduced
    print(np.max(np.linalg.pinv(A_reduced,rcond=my_rcond)))

    print(F_est[-1])

    return np.sign(F_est[-1])*F_est[:-1], timeavg_freq/T, unvisited_genotypes
    # return F_est, timeavg_freq/T

# function to calculate Pearson r values at various rare frequency cutoffs
def get_correlation_values(F_est,F_real,timeavg_freq,rare_cutoffs):
    timeavg_freq = timeavg_freq[F_real == F_real]
    F_est = F_est[F_real == F_real]
    F_real = F_real[F_real == F_real]
    

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
    nlogrcond = args.nlogrcond
    filedir = args.dir

    filesuffix = ''
    datapath = filedir + '/' + modeltype + '_freq_timeseries' + filesuffix + '.csv'
    mutationpath = filedir + '/' + modeltype + '_M.csv'
    fitnesspath = filedir + '/' + modeltype + '_F_real.csv'
    outpath = filedir + '/' + modeltype + '_processed' + filesuffix + '.pkl'

    # get time series data, data assumed to have shape [Nvariants, T]
    freq_timeseries = np.loadtxt(datapath,delimiter=',')
    T = freq_timeseries.shape[1]
    Nvariants = freq_timeseries.shape[0]

    # get mutation matrix
    M = 1e-3 * -np.eye((freq_timeseries.shape[0]))

    # get real fitnesses
    F_real = np.loadtxt(fitnesspath,delimiter=',')

    # get estimated fitnesses, timeavg_freq
    F_est_reduced, timeavg_freq, unvisited_genotypes = estimate_fitness(freq_timeseries,M,nlogrcond)
    F_real_reduced = np.delete(F_real, unvisited_genotypes)
    timeavg_freq_reduced = np.delete(timeavg_freq, unvisited_genotypes)
    F_est = np.zeros((len(timeavg_freq)))
    F_est[np.delete(np.arange(len(timeavg_freq)),unvisited_genotypes)] = F_est_reduced
    F_est[unvisited_genotypes] = np.nan

    # get Pearson r at various cutoffs
    rare_cutoffs = 10 ** np.arange(-5,-0,0.25)
    correlation_data = get_correlation_values(F_est_reduced,F_real_reduced,timeavg_freq_reduced,rare_cutoffs)

    # save outputs to file
    data = {
            'F_real': F_real,
            'F_real_reduced': F_real_reduced,
            'F_est': F_est,
            'F_est_reduced': F_est_reduced,
            'timeavg_freq': timeavg_freq,
            'timeavg_freq_reduced': timeavg_freq_reduced,
            'unvisited_genotypes': unvisited_genotypes,
            'rare_cutoffs': rare_cutoffs,
            'correlation_data': correlation_data
            }
    
    with open(outpath,'wb') as file:
        pickle.dump(data,file)
    
    timeavg_and_F = np.vstack((timeavg_freq,F_est)).T
    np.savetxt(modeltype + '_F.csv',timeavg_and_F,delimiter=',')