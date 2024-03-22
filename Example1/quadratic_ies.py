# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:00:55 2022

@author: syang
"""


import numpy as np
import matplotlib.pyplot as plt
from IES_lib import IES

#
# Define a problem: fit the line y = n1*x^2 + n2*x + n3 using observation points

#
np.random.seed(0)
def g(n1, n2, n3, loc):
    y = n1*loc**2 + n2*loc + n3
    return y

def costf(n1, n2, n3, loc, d_obs):
    y = n1*loc**2 + n2*loc + n3
    rmse = (np.sum((y - d_obs) ** 2) / len(d_obs)) ** 0.5
    return rmse
    
#
def get_ensemble_d(ensemble_m, Nd, Ne):
    ensemble_d = np.zeros((Ne, Nd))
    en_rmse = np.zeros(Ne)
    for i in range(ensemble_m.shape[0]):
        oned = g(ensemble_m[i][0], ensemble_m[i][1], ensemble_m[i][2], loc)
        ensemble_d[i, :] = oned
        en_rmse[i] = costf(ensemble_m[i][0], ensemble_m[i][1], ensemble_m[i][2], loc, d_obs)
    return ensemble_d, en_rmse

def plot_ensemble(m_cur, axs, axid):
    x = np.linspace(-10, 10, 15)
    true_sol = g(1, 3, 2, x)
    row = axid // 5
    col = axid % 5
    axs[row][col].plot(x, true_sol, color = 'blue', linewidth = 2.0)
    for i in m_cur:
        y = g(i[0], i[1], i[2], x)
        axs[row][col].plot(x, y, color = 'black', linewidth = 0.3)
        axs[row][col].set_title('iteration{}'.format(axid), fontsize = 20)

#
# Generate observations
n1 = 1; n2 = 3; n3 = 2 # true solution
loc = np.linspace(-10, 10, 15) # x values
d_obs_true = g(n1, n2, n3, loc) # y values on the true function

# Add noise to obervations
d_obs = d_obs_true + np.random.normal(0.0, 0.25, len(d_obs_true))

# Plot true function and observation data
fig, ax = plt.subplots(1)
ax.plot(loc, d_obs_true)
ax.scatter(loc, d_obs)
fig.savefig('Observation_data.png', dpi = 500)


# Give bounds for n1, n2, n3, which are used in ES-LM method
var_bound = [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
var_bound = np.array(var_bound)

# Plot realizations produced by the ES-LM
fig, axs = plt.subplots(2,5, figsize = (40,15))
fig.suptitle('ES-LM', fontsize = 30)

# Initialize ES-LM
# N_realization: number of realizations.
# Cd: Covariance of observation error, assumed a diagonal matrix.
# var_bound: 2D_arry, bounding values of each parameter (n1, n2, n3).
# Observation: observation data, 1D_array
ies = IES(N_realization = 50, Cd = np.diag(np.ones(len(d_obs))*0.5), 
          var_bound = var_bound, Observations = d_obs)

# Solve iteratively and interactively
rmse_coll = [] # record RMSE, not required by the ES-LM

for iter in range(0, 10):
    # Ask for an ensemble of parameters
    # para: 2D_array of the ensemble with size (Ne, Nm).
    # Ne: number of realizations.
    # Nm: number of parameters.
    para = ies.ask(n_iter = iter)
    
    #
    plt.figure(iter+3)
    plot_ensemble(para, axs=axs, axid=iter)
    print('Ensemble mean = ', np.mean(para, axis = 0))
    
    # Given the "para", calculate the ensemble of function values at observation location
    # data_simu: 2D_array of the ensemble with size (Ne, Nd).
    # Nd: number of observation data.
    data_simu, rmse = get_ensemble_d(ensemble_m = para, Nd = ies.Nd, Ne = ies.Ne) 
    rmse_coll.append(rmse.mean())
    
    # Tell the ensemble of function values
    ies.tell(data_simu = data_simu, n_iter = iter)
    
#
fig.savefig('ES-LM.png', dpi = 500)


# Plot convergence of RMSE
fig, ax = plt.subplots(1)
ax.plot(np.arange(len(rmse_coll)), np.array(rmse_coll), marker = 'o')
fig.savefig('RMSE.png', dpi = 500)
