import glob
import os
import numpy as np
cores = 1
nodes = 1
days = 0
hours = 5
minutes = 0
memory = 4000 # in MB
array_start = 0
array_end = 0
NUM_JOBS = 100
print('There are',NUM_JOBS,'trials to submit.')
T = 1000
N = 1000
L = 10
mu = 0.01
out_dir = 'data'
model = 'serialdilution'
for i in range(48,53):
    with open('run.sh','w') as f:
        filename = model + '_trial' + str(i)
        f.write('#!/bin/bash\n')
        f.write('#SBATCH -n ' + str(cores) + '\n')
        f.write('#SBATCH -N ' + str(nodes) + '\n')
        f.write('#SBATCH -C centos7\n')
        f.write('#SBATCH -t ' + str(days) + '-' + str(hours) + ':' + str(minutes) + '\n')
        f.write('#SBATCH -p sched_mit_hill\n')
        f.write('#SBATCH --mem-per-cpu=' + str(memory) + '\n')
        # f.write('#SBATCH --mail-type=END\n')
        f.write('#SBATCH -o ' + filename + '_%j.out\n')
        f.write('#SBATCH -e ' + filename + '_%j.err\n')
        # f.write('source ../fe_venv/bin/activate' + '\n')
        f.write('python3 ' + model + '.py --T ' + str(T) + ' --L ' + str(L) + ' --N ' + str(N) + ' --mu ' + str(mu) + ' --dir \'' + out_dir + '\' ' + ' --trial ' + str(i) + '\n')
        f.write('python3 fitness_estimation.py --model \'' + model + '\' --T ' + str(T) + ' --L ' + str(L) + ' --N ' + str(N) + ' --mu ' + str(mu) + ' --dir \'' + out_dir + '\' ' + ' --trial ' + str(i) + '\n')
    os.system('sbatch run.sh')
    print('Submitting job: ' + filename + '\n')