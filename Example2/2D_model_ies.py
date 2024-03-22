# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:24:31 2022

@author: syang
"""


import numpy as np
from IES_lib import IES
import matplotlib.pyplot as plt
import os
import csv
import flopy.utils.binaryfile as bf
import subprocess

# Problem: estimate parameters (K) of a MODFLOW-USG groundwater model, given 
# observation data.
# Number of parameters: 25.
# Number of observation data: 34.

#
def write_hk(HK, offset = 10):
    hk_ls = []
    for i in range(len(HK)//offset):
        hk_ls.append(10**HK[i*offset:(i+1)*offset])
    with open('HK.dat', "w",newline='') as output:
        writer = csv.writer(output,delimiter=' ')
        writer.writerows(hk_ls)

def run_model(HK, obs_ind):
    write_hk(HK=HK)
    runsim=subprocess.run('mfusg_x64.exe 2D_model.mfn',timeout=60*500)
    hed = bf.HeadUFile('2D_model.hed', precision='single')
    x=hed.get_data(idx=0)
    sim_head = x[0]
    obs = sim_head[obs_ind]
    return obs

def get_ensemble_d(ensemble_m, Nd, Ne, hk_true, para_ind, obs_ind):
    ensemble_d = np.zeros((Ne, Nd))
    ensemble_rmse = np.zeros(Ne)
    for i in range(ensemble_m.shape[0]):
        HK = hk_true.copy()
        HK[para_ind] = ensemble_m[i]
        oned = run_model(HK = HK, obs_ind = obs_ind)
        ensemble_d[i, :] = oned
        ensemble_rmse[i] = np.sqrt(np.sum((obs_true - oned)**2) / len(obs_true)) # rmse
    return ensemble_d, ensemble_rmse


#
np.random.seed(0)

# True K field, used for the cells where K values are assumed known. This is only
# for the toy problem. In practice, don't consider this.
hk_true = np.loadtxt('hk_true.txt')

# Index of cells with unknown K
para_ind = np.loadtxt('para_ind.txt').astype('int')

# Index of cells that have observation data
obs_ind = np.loadtxt('obs_ind.txt').astype('int')

# Observation data, noise added
obs_true = np.loadtxt('obs_true.txt')

#
rdir = os.getcwd()

#
# Give bounds for model parameters (K), which are used in ES-LM method
var_bound = [[-5.0, 5.0]]*len(para_ind)
var_bound = np.array(var_bound)

#
# Initialize ES-LM
# N_realization: number of realizations.
# Cd: Covariance of observation error, assumed a diagonal matrix.
# var_bound: 2D_arry, bounding values of each parameter (n1, n2, n3).
# Observation: observation data, 1D_array
ies = IES(N_realization = 100, Cd = np.diag(np.ones(len(obs_ind))*0.5), var_bound = var_bound, Observations = obs_true)

# Solve iteratively and interactively
rmse_coll=[] # record RMSE, not required by the ES-LM

for niter in range(0, 20):
    print('Iteration ', niter)
    os.chdir('2D_usg_model')
    
    # Ask for an ensemble of parameters
    # para: 2D_array of the ensemble with size (Ne, Nm).
    # Ne: number of realizations.
    # Nm: number of parameters.
    para = ies.ask(n_iter = niter)
    
    # Given the "para", run the model, and 
    # get the ensemble of model output at observation location
    # data_simu: 2D_array of the ensemble with size (Ne, Nd).
    # Nd: number of observation data.
    data_simu, values = get_ensemble_d(ensemble_m = para, Nd = ies.Nd, Ne = ies.Ne, hk_true=hk_true, 
                               para_ind=para_ind, obs_ind=obs_ind) 
    rmse = np.mean(values)
    rmse_coll.append(rmse)
    
    # Tell the ensemble of model output
    ies.tell(data_simu = data_simu, n_iter = niter)
    #
    os.chdir(rdir)
    #np.savetxt('rmse_ies.txt', np.array(rmse_coll), fmt='%.3f')

#
# Plot convergence of RMSE
fig, ax = plt.subplots(1)
ax.plot(np.arange(len(rmse_coll)), np.array(rmse_coll), marker = 'o')
fig.savefig('RMSE.png', dpi = 500)
