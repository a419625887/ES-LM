# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:03:16 2022

@author: syang
"""


# This is a module for LM based Iterative Ensemble Smoother.
# All of codes are based on the paper of Ma and Bi (2019).

import numpy as np
from numpy.linalg import inv
from scipy.linalg import fractional_matrix_power
from scipy.linalg import norm
from scipy.stats import qmc

class IES():
    def __init__(self, N_realization, Cd, var_bound, Observations):
        #  Initialization
        np.random.seed(0)
        self.Ne = N_realization
        self.Cd = Cd
        self.Cdinv = inv(Cd)
        self.var_bound = var_bound
        self.n_var = len(var_bound)
        self.d_obs = Observations
        self.Nd = len(self.d_obs)
        #
        self.do = self.perturb_obs(cov = Cd) # perturb observations
        #
        m_pri = self.lhs_sampling()
        #m_pri = np.zeros((self.Ne, self.n_var))
        #for i in range(self.n_var):
        #    m_pri[:, i] = np.random.uniform(var_bound[i, 0], var_bound[i, 1], size = self.Ne)
        #pri_mean =np.array([2,3])
        #pri_cov = np.diag(np.array([3.0**2, 3.0**2]))
        #m_pri = np.random.multivariate_normal(mean = pri_mean, cov = pri_cov, size = self.Ne) # variable values from prior distribution
        #
        self.m_cur = m_pri.copy() # ensemble of m_i_j
        
    #
    def lhs_sampling(self):
        sampler = qmc.LatinHypercube(d=self.n_var)
        sample = sampler.random(n=self.Ne)
        log_field = np.zeros((self.Ne, self.n_var))
        for j in range(self.n_var):
            log_field[:, j] = self.var_bound[j][0] + sample[:, j] * (self.var_bound[j][1] - self.var_bound[j][0])
        #
        return log_field
    #
    def perturb_obs(self, cov):
        do = np.zeros((self.Ne, self.Nd))
        for i in range(self.Ne):
            do[i] = self.d_obs + np.random.normal(loc = 0.0, scale = cov[0][0]**0.5, size = self.Nd)
        #for i in range(self.Ne):
        #    do[i] = self.d_obs + np.random.multivariate_normal(mean = np.array([0.0]*self.Nd), cov = cov, size = 1)
        #
        return do
    #
    def get_Cmd(self):
        m_bar = np.mean(self.m_cur, axis=0)
        d_bar = np.mean(self.d_cur, axis=0)
        Cmd = np.zeros((self.n_var, self.Nd))
        for i in range(self.Ne):        
            Cmd += np.matmul((self.m_cur[i] - m_bar).reshape((self.n_var, 1)), (self.d_cur[i] - d_bar).reshape((1, self.Nd)))
        #
        Cmd = Cmd/(self.Ne - 1)
        return Cmd
    #
    def get_Cdd(self):
        d_bar = np.mean(self.d_cur, axis=0)
        Cdd = np.zeros((self.Nd, self.Nd))
        for i in range(self.Ne):
            Cdd += np.matmul((self.d_cur[i] - d_bar).reshape((self.Nd, 1)), (self.d_cur[i] - d_bar).reshape((1, self.Nd)))
        #
        Cdd = Cdd/(self.Ne - 1)
        return Cdd
    #
    def get_m_next(self):
        m_next = np.zeros((self.Ne, self.n_var))
        med = np.matmul(self.Cmd, inv(self.Cdd + self.alpha * self.Cd))
        for i in range(self.Ne):
            onemember = self.m_cur[i].reshape((self.n_var, 1)) + np.matmul(med, (self.do[i] - self.d_cur[i]).reshape((self.Nd, 1)))
            #print(onemember)
            m_next[i, :] = onemember.reshape((self.n_var,))
        #
        trun_m = self.truncate_m(m_raw = m_next) # truncate m in case there are values out of bounds.
        return trun_m
    #
    def truncate_m(self, m_raw):
        # Truncate updated parameters that are out of the specified bounds
        trun_m = np.zeros((self.Ne, self.n_var))
        for i in range(self.n_var):
            u_b = self.var_bound[i][1]
            l_b = self.var_bound[i][0]
            for j in range(self.Ne):
                if m_raw[j][i] < l_b:
                    trun_m[j][i] = l_b
                elif m_raw[j][i] > u_b:
                    trun_m[j][i] = u_b
                else:
                    trun_m[j][i] = m_raw[j][i]
        #
        return trun_m
    #
    def get_O_bar(self):
        O_bar = 0.0
        for n in range(self.Ne):
            med = np.matmul((self.d_cur[n, :] - self.d_obs), self.Cdinv)
            O_bar += np.matmul(med, (self.d_cur[n, :] - self.d_obs).reshape((self.Nd, 1))) * (1/(2.0*self.Nd))
        #
        O_bar = O_bar[0]/self.Ne
        return O_bar
    #
    def get_ensemble_O(self, ensemble_d):
        O = np.zeros(self.Ne)
        for i in range(self.Ne):
            med1 = np.matmul((self.do[i] - ensemble_d[i]).reshape((1, self.Nd)), self.Cdinv)
            med2 = np.matmul(med1, (self.do[i] - ensemble_d[i]).reshape((self.Nd, 1)))
            O[i] = med2[0]*0.5
        #
        return O
    #
    def get_ensemble_L(self, ensemble_d):
        L = np.zeros(self.Ne)
        med1 = np.matmul(fractional_matrix_power(self.Cd, 0.5), inv(self.Cdd + self.alpha*self.Cd))
        for i in range(self.Ne):
            med2 = np.matmul(med1, (self.do[i] - ensemble_d[i]).reshape((self.Nd, 1)))
            L[i] = (self.alpha**2) * (norm(med2)**2)
        #
        return L
    #
    def get_ensemble_gamma(self):
        en_gamma = np.zeros(self.Ne)
        for i in range(self.Ne):
            med = 1 - (2*self.rou[i] - 1)**3
            if (1/3) >= med:
                en_gamma[i] = self.gamma * (1/3)
            else:
                en_gamma[i] = self.gamma * med
        #
        return en_gamma
    #
    def ask(self, n_iter):
        if n_iter == 0:
            return self.m_cur
        else:
            self.Cmd = self.get_Cmd()
            self.Cdd = self.get_Cdd()
            self.m_next = self.get_m_next() # ensmeble of m_i+1_j
            return self.m_next
    #
    def tell(self, data_simu, n_iter):
        if n_iter == 0:
            self.d_cur = data_simu
            self.O_bar = self.get_O_bar()
            self.gamma = 1.0
            self.alpha = self.gamma * self.O_bar
        else:
            self.d_next = data_simu
            self.O_cur = self.get_ensemble_O(ensemble_d = self.d_cur) # ensemble of O_i_j(m_i_j)
            self.O_next = self.get_ensemble_O(ensemble_d = self.d_next) # ensemble of O_i_j(m_i+1_j)
            self.L_cur = self.O_cur.copy() # ensemble of L_i_j(m_i_j)
            self.L_next = self.get_ensemble_L(ensemble_d = self.d_cur) # ensemble of L_i_j(m_i+1_j)
            self.O_bar = self.get_O_bar() # O_bar_i
            #
            self.rou= np.zeros(self.Ne) # ensemble of rou_j
            for i in range(self.Ne):
                self.rou[i] = (self.O_cur[i] - self.O_next[i])/(self.L_cur[i] - self.L_next[i])
            #
            self.en_gamma = self.get_ensemble_gamma() # ensemble of gamma_i_j
            self.en_alpha = self.en_gamma * self.O_bar # ensemble of alpha_i_j
            self.gamma = np.median(self.en_gamma) # update gamma for next iteration
            self.alpha = np.min(np.array([self.alpha, np.median(self.en_alpha)])) # update alpha for next iteration
            #
            self.do = self.perturb_obs(cov=self.alpha * self.Cd) # perturb observations for next iteration
            self.m_cur = self.m_next.copy()
            self.d_cur = self.d_next.copy()
    #
