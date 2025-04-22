# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:04:23 2016

@author: matt
"""

#%%
import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import torch

error_list = []
def GaussianMatrix(X,Y,sigma):
    size1 = X.size()
    size2 = Y.size()
    G = (X*X).sum(-1)
    H = (Y*Y).sum(-1)
    Q = G.unsqueeze(-1).repeat(1,size2[0])
    R = H.unsqueeze(-1).T.repeat(size2[0],1)
    
    
    H = Q + R - 2*X@(Y.T)
    H = torch.exp(-H/2/sigma**2)
    
    
    return H

def CSD_4(x1,x2,y1,y2,sigma = 1): # conditional cs divergence
    x1 = torch.tensor(x1)
    x2 = torch.tensor(x2)
    y1 = torch.tensor(y1)
    y2 = torch.tensor(y2)
    
    
    K1 = GaussianMatrix(x1,x1,sigma)
    K2 = GaussianMatrix(x2,x2,sigma)
    
    L1 = GaussianMatrix(y1,y1,sigma)
    L2 = GaussianMatrix(y2,y2,sigma)
    
    K12 = GaussianMatrix(x1,x2,sigma)
    L12 = GaussianMatrix(y1,y2,sigma)
    
    K21 = GaussianMatrix(x2,x1,sigma);
    L21 = GaussianMatrix(y2,y1,sigma);

    H1 = K1*L1
    self_term1 = (H1.sum(-1)/((K1.sum(-1))**2)).sum(0)
    
    H2 = K2*L2
    self_term2 = (H2.sum(-1)/((K2.sum(-1))**2)).sum(0)
    
    H3 = K12*L12;
    cross_term1 = (H3.sum(-1)/((K1.sum(-1))*(K12.sum(-1)))).sum(0)
    
    H4 = K21*L21;
    cross_term2 = (H4.sum(-1)/((K2.sum(-1))*(K21.sum(-1)))).sum(0)
    
    cs1 = -2*torch.log2(cross_term1) + torch.log2(self_term1) + torch.log2(self_term2)
    cs2 = -2*torch.log2(cross_term2) + torch.log2(self_term1) + torch.log2(self_term2)
    
    
    return ((cs1+cs2)/2).item()

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_d_kullback_leibler

from ite.cost.x_factory import co_factory
from ite.cost.x_analytical_values import analytical_value_d_kullback_leibler
cost_name = 'BDKL_KnnK'
distr = 'normal'
co = co_factory(cost_name, mult=True)

# import dill
#%% parameters

ENVIRONMENT = 'MountainCarContinuous-v0'
WATCH = True

# end when goal/episode is complete?
GOAL_DONE = False 

UPDATE_BATCH = False
Q_REPLAY_MOD = 10
Q_REPLAY_BATCH = 25

# use 'dtg' or 'q' to select actions
DTG_Q_FLAG = 'dtg'

# DIVERGENCE TYPE: EUCLIDEAN = 0, CS = 1
DIV_TYPE = 0

# Number of nearest neighbors to use for transition model
KNN_N = 20

N_STEPS = 50000
RAND_ITERS = 250
#RAND_ITERS = N_STEPS

KERNEL_SIZE_STATE = .001
KERNEL_SIZE_REWARD = 0.1

SIM_SIZE_STATE = .005
SIM_SIZE_ACTION = .1

DTG_KERNEL_SIZE_STATE = 0.1
DTG_KERNEL_SIZE_ACTION = 0.1

DTG_KERNEL_SIZE_ENT = 10000.
DTG_KERNEL_SIZE_MEAN = 10000.
DTG_QUANT_EPS = 0.000001


Q_KERNEL_SIZE_STATE = .1
Q_KERNEL_SIZE_ACTION = .1


#ACTION_LIST = np.arange(-1.,1.1,.1)
ACTION_LIST = np.array([-1,0,1])

MAX_DIVERG = 25.


ALPHA = 0.1 # learning rate

BATCH_ALPHA = 3.0

GAMMA = 0.9 # discount factor

KAPPA = 2.

ZERO_EPS = 10**-6

INIT_DTG = MAX_DIVERG / ( 1-GAMMA) * KAPPA
INIT_Q = MAX_DIVERG / (1-GAMMA) * KAPPA

XLIMS = (-1.2, .5)
YLIMS = (-.07, .07)


#%%
class TransitionKernelModel(object):
    
    def __init__(self, sim_size_state, sim_size_action, kernel_size, kernel_size_reward, 
                 eps, div_type, n_knn, state_dim, action_dim, action_list, T_size, max_diverg):
        self.sim_size_state = sim_size_state
        self.sim_size_action = sim_size_action
        self.kernel_size = kernel_size
        self.kernel_size_reward = kernel_size_reward
        self.eps = eps
        self.div_type = div_type        
        self.n_knn = n_knn        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if action_list.ndim == 2:
            self.action_list = action_list
        elif action_list.ndim == 1:
            self.action_list = action_list[:,None]
        
        self.T_size = T_size # room to reserve for dictionary       
        
        self.max_diverg = max_diverg
        
        self.xa = None
        self.Traw = np.zeros((T_size, state_dim*2 + action_dim + 1))  # reserve room for T
        self.T = None # transition dictionary
        self.Tnn = None # reduced T
        self.sim_state_vec = None
        self.sim_action_vec = None
        self.Vmat = None
        
        self.last_pdf_x = None
        self.last_pdf_a = None
        self.last_pdf_T = None
        self.last_pdf_sim_norm = None
        
        self.step = 0

        self.diverg_list = []
        
        
    def compute_pdf(self, x, y, a, T=None, sim_size_state=None, sim_size_action=None, kernel_size=None):
        
        if T is None:
            T = self.T
        if sim_size_state is None:
            sim_size_state = self.sim_size_state
        if sim_size_action is None:
            sim_size_action = self.sim_size_action
        if kernel_size is None:
            kernel_size = self.kernel_size
        
        xdim = self.state_dim
        adim = self.action_dim        
        
        # if x is being evaluated a bunch of times, 
        # don't bother finding the nearest neighbors over and over again
        if np.array_equal(x, self.last_pdf_x):
            T = self.last_pdf_T
        else:
            nbrs = NearestNeighbors(n_neighbors=min(self.n_knn,T.shape[0])).fit(T[:,0:xdim])
            _, knnidx = nbrs.kneighbors(x.reshape(1,-1))
            T = T[knnidx.squeeze(),:]
            self.last_pdf_T = T
               
        # if x AND a stays the same don't, recompute sim_norm   
        if np.array_equal(a,self.last_pdf_a) and np.array_equal(x, self.last_pdf_x):
            sim_norm = self.last_pdf_sim_norm
        else:
            sim_state  = np.exp(-np.sum( (x-T[:,0:xdim])**2, 1) / sim_size_state**2 ) 
            sim_action = np.exp(-np.sum( (a-T[:,(xdim*2):(xdim*2+adim)])**2, 1 ) / sim_size_action**2)
            sim = sim_state * sim_action
            sim_norm = sim / np.sum(sim,0)
            
            self.last_pdf_sim_norm = sim_norm
        
        
        kernel_eval = np.sum(sim_norm*np.exp(-np.sum(((y-x) - (T[:,xdim:xdim*2]-T[:,0:xdim]))**2,1)/kernel_size))
        
        self.last_pdf_x = x
        self.last_pdf_a = a
        
        return kernel_eval
    
    def update_transition_model(self, state, new_state, action, reward):
        self.Traw[self.step,:] = np.concatenate([state, new_state, action, [reward]])
        self.step += 1
        if self.step > self.T_size:
            print('transition model room exceeded')
        self.T = self.Traw[:self.step,:]
    
    ## update the parameters of transition model dependent on position 
    def _update_local_params(self, x, a=None, n_knn=None):
        
        if n_knn is None:
            n_knn = self.n_knn
        
        xdim = self.state_dim
        adim = self.action_dim
        
        if a is None:
            xa = x
            Ta = self.T[:,0:xdim]
        else:
            xa = np.concatenate([x,a])
            self.xa = xa        
            Ta = np.hstack([self.T[:,0:xdim],self.T[:,xdim*2:xdim*2+adim]])
        
        # compute reduced T using nearest neighbors
        
        nbrs = NearestNeighbors(n_neighbors=min(int(n_knn),self.T.shape[0])).fit(Ta)
        _, knnidx = nbrs.kneighbors(xa.reshape(1,-1))
        knnidx = np.sort(knnidx)
        
        Tnn = self.T[knnidx.flatten(),:]

        # randomly permute the rows of Tnn
        #Tlen = Tnn.shape[0]
        #Tinds = np.random.permutation(Tlen)
        #Tnn = Tnn[Tinds,:]
        
        # save Tnn
        self.Tnn = Tnn        
               
        # compute similarity vector
        self.sim_state_vec = np.exp(-np.sum( (x-Tnn[:,0:xdim])**2, 1) / self.sim_size_state**2 )
        
        # compute Vmat
        TT = Tnn[:,xdim:xdim*2]-Tnn[:,0:xdim]
        RR = Tnn[:,xdim*2+adim]       
        self.Vmat = np.exp( -((TT[:,np.newaxis,:]-TT)**2).sum(-1) / self.kernel_size**2 ) * \
                    np.exp( -((RR[:,np.newaxis] - RR)**2) / self.kernel_size_reward**2  ) 
        
    
    def compute_divergence(self, x, a):
        
        xa = np.concatenate([x,a])        
        if not np.array_equal(xa, self.xa):
            self._update_local_params(x,a)
        
        Tlen = self.Tnn.shape[0]        
        xdim = self.state_dim
        adim = self.action_dim

        sim_action_vec = np.exp(-np.sum( (a-self.Tnn[:,(xdim*2):(xdim*2+adim)])**2, 1 ) / self.sim_size_action**2)
        
        sim_vec = self.sim_state_vec * sim_action_vec        
        
        sim1 = sim_vec[:Tlen//2]
        #print(Tlen)
        sim_norm1 = np.nan_to_num(sim1 / np.sum(sim1,0))
        sim_norm1 = sim_norm1 * (sim_norm1 > self.eps)
                
        sim2 = sim_vec[Tlen//2:]
        sim_norm2 = np.nan_to_num(sim2 / np.sum(sim2,0))
        sim_norm2 = sim_norm2 * (sim_norm2 > self.eps)
        
        N1 = sim_norm1.shape[0]
        N2 = sim_norm2.shape[0]
    



        
        x_old = self.Tnn[:Tlen//2,0:xdim]
        y_old = self.Tnn[:Tlen//2,xdim:xdim*2]
        a_old = self.Tnn[:Tlen//2,(xdim*2):(xdim*2+adim)]
        
        x_new = self.Tnn[Tlen//2:Tlen//2*2,0:xdim]
        y_new = self.Tnn[Tlen//2:Tlen//2*2,xdim:xdim*2]
        a_new = self.Tnn[Tlen//2:Tlen//2*2,(xdim*2):(xdim*2+adim)]
        
        x1 = np.concatenate((x_old,a_old),-1)
        x2 = np.concatenate((x_new,a_new),-1)
        
        csd_our = CSD_4(x1,x2,y_old,y_new,sigma = 1)
        
        if x1.shape[0]>x1.shape[1]:
            csd_our = co.estimation(np.concatenate((x1,y_old),1), np.concatenate((x2,y_new),1))- co.estimation(x1, x2)
        
        divs = [csd_our, csd_our]#D_euc
        
        if np.isnan(csd_our) or np.isnan(csd_our):
            print(divs)
        
        return divs
    
        
    

    def compute_entropy(self, x, a=None):
        # can compute Vmat and the first part (state) of sim1 once, compute action part many times
        
        
        if a is None:
            a = self.action_list
            self._update_local_params(x,n_knn=self.n_knn*self.action_list.shape[0]/4)
        else:
            xa = np.concatenate([x,a]) 
            if not np.array_equal(xa, self.xa):
                self._update_local_params(x,a)
            
            
        T = self.Tnn
  
        xdim = self.state_dim
        adim = self.action_dim        
        
        Ta = self.Tnn[:,(xdim*2):(xdim*2+adim)]
        
        sim_state = self.sim_state_vec[:,None]
        sim_action = np.exp(-np.sum((Ta[:,np.newaxis,:]-a)**2,2) / self.sim_size_action)

        sims = sim_state * sim_action
        sims_norm = np.nan_to_num(sims / np.sum(sims,0))
        sims_norm = sims_norm * (sims_norm > self.eps)
        
        N = sims_norm.shape[0]
        
        entropies = np.diag(np.dot(np.dot(sims.T,self.Vmat),sims)) / N**2
        
        pdf_mean = np.nan_to_num(np.sum((T[:,xdim:xdim*2] - T[:,0:xdim]).T * sims_norm.T[:,np.newaxis,:],2))
        
        if np.isnan(pdf_mean).any():
            print(pdf_mean)

        
        return entropies, pdf_mean
        
    def reset(self):
        print('resetting transition model...')
        
        self.xa = None
        self.Traw = np.zeros((self.T_size, self.state_dim*2 + self.action_dim + 1))  # reserve room for T
        self.T = None
        self.Tnn = None
        self.sim_state_vec = None
        self.sim_action_vec = None
        self.Vmat = None
        
        self.last_pdf_x = None
        self.last_pdf_a = None
        self.last_pdf_T = None
        self.last_pdf_sim_norm = None
        
        self.step = 0

        self.diverg_list = []

        
    def load_table(self, Traw):
        self.T_size = Traw.shape[0] + N_STEPS
        self.Traw = np.zeros((self.T_size, self.state_dim*2 + self.action_dim + 1))
        self.Traw[0:Traw.shape[0],:] = Traw
        self.step = Traw.shape[0]+1
        self.T = self.Traw[:self.step,:]


#%%
class DtgKlmsModel(object):
    def __init__(self, alpha, gamma, dtg_kernel_size_state, dtg_kernel_size_action, 
                 dtg_kernel_size_ent, dtg_kernel_size_mean, 
                 state_dim, action_dim, init_dtg, eps, quant_eps, n_steps):
                     
        self.alpha = alpha
        self.gamma = gamma
        self.dtg_kernel_size_state = dtg_kernel_size_state
        self.dtg_kernel_size_action = dtg_kernel_size_action
        self.dtg_kernel_size_ent = dtg_kernel_size_ent
        self.dtg_kernel_size_mean = dtg_kernel_size_mean
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.init_dtg = init_dtg
        self.eps = eps
        self.quant_eps = quant_eps
        self.n_steps = n_steps
              
        self.dtg_ks_vec = np.hstack([np.tile(dtg_kernel_size_state,state_dim), 
                                     np.tile(dtg_kernel_size_action,action_dim),
                                     np.tile(dtg_kernel_size_ent,1),
                                     np.tile(dtg_kernel_size_mean,state_dim)])
                                     

        self.dtg_error_table = np.zeros((n_steps, state_dim+action_dim+1+state_dim+1))
                
        self.step = 0
    
    ## dtg klms model 
    ## sim_kernel_eval_dtg_klms_cts
    def compute_dtg(self, x, y, T=None, ks_vec=None, 
                    init_dtg=None, eps=None):
        
        if T is None:
            T = self.dtg_error_table[0:self.step,:]
        if ks_vec is None:
            ks_vec = self.dtg_ks_vec
        if init_dtg is None:
            init_dtg = self.init_dtg
        if eps is None:
            eps = self.eps
        # x = state-action-div_params to compute kernel_pre, div_params = mean/entropy
        # y = list of state-actions to compute kernel_post_list
        # T = state-action ERROR TABLE
        # ks_vec = kernel size vector---one kernel size for each state-action dimension
        # init_dtg = value to initialize dtg at
        dim = y.shape[1]
        
        if x is not None:
            xk = x * (np.abs(x) > eps)
            xk = xk / ks_vec
        yk = y / ks_vec
        Tk = T / np.concatenate([ks_vec,[1.]])    
        
        if x is None:
            kernel_pre = None
        else:
            if x.ndim == 1:
                kernel_pre = init_dtg + np.sum(Tk[:,dim] * np.exp(-np.sum( (xk-Tk[:,0:dim])**2, 1) ))
            elif x.ndim ==2:
                kernel_pre = init_dtg + np.sum(Tk[:,dim] * np.exp(-np.sum((xk[:,np.newaxis,:] - Tk[:,0:dim])**2,-1)),1)
        
        kernel_post_list = init_dtg + np.sum(Tk[:,dim] * np.exp(-np.sum((yk[:,np.newaxis,:] - Tk[:,0:dim])**2,-1)),1)
        
        if np.isnan(kernel_post_list).any():
            print(kernel_post_list)
        
        return [kernel_pre, kernel_post_list]

    
    def update_dtg(self, state_action, state_action_list, diverg):
                    
        [dtg_pre,dtg_post_list] = self.compute_dtg(state_action, state_action_list)
        
        #print(diverg)
        ## update dtg error using TD equation      
        dtg_error = self.alpha * ((diverg + self.gamma*np.max(dtg_post_list)) - dtg_pre)
        #print(dtg_error)
        if np.isnan(dtg_error):
            print(dtg_error)
        
        dists = np.linalg.norm(self.dtg_error_table[:self.step,:-1] - state_action, axis=1)
        if 0:#(dists < self.quant_eps).any():
            self.dtg_error_table[np.argmin(dists),-1] += dtg_error
        else:
            self.dtg_error_table[self.step,:] = np.concatenate([state_action, [dtg_error]])
            self.step += 1
            
    
    def update_dtg_batch(self, sa_pre, sa_post, diverg):
        diverg = 0
        # sa_pre = 2-d array (batch_size, state_action_dim)
        # sa_post = 2-d array (action_list size * batch_size, state_action_dim)
        # diverg = vector (batchsize)
        # dtg_error = vector (batchsize)        

        aa = self.dtg_error_table[:,:-1]
        nbrs = NearestNeighbors(n_neighbors=1).fit(aa)
        _, knnidx = nbrs.kneighbors(sa_pre)
    
        batchsize = sa_pre.shape[0]

        [dtg_pre,dtg_post1] = self.compute_dtg(sa_pre, sa_post)
        dtg_post = dtg_post1.reshape(batchsize,-1)
        
        dtg_error = self.alpha * ( (diverg +  (self.gamma*np.max(dtg_post,axis=1)) - dtg_pre) )
        
        self.dtg_error_table[knnidx,-1] += dtg_error[:,None] / batchsize
        
        
    def reset(self):
        print('resetting dtg...')
        self.dtg_error_table = np.zeros((self.n_steps, self.state_dim+self.action_dim+1+self.state_dim+1))
        
        self.step = 0



class QKlmsModel(object):
    def __init__(self, alpha, gamma, q_kernel_size_state, q_kernel_size_action, state_dim,
                 action_dim, init_q, eps, n_steps):
        
        self.alpha = alpha
        self.gamma = gamma
        self.q_kernel_size_state = q_kernel_size_state
        self.q_kernel_size_action = q_kernel_size_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.init_q = init_q
        self.eps = eps
        self.n_steps = n_steps


        self.q_ks_vec = np.hstack([np.tile(q_kernel_size_state,state_dim), 
                                   np.tile(q_kernel_size_action,action_dim)])
        self.q_error_table = np.zeros((n_steps, state_dim+action_dim+1))
        
        self.step = 0
        
    def compute_q(self, x, y, T=None, ks_vec=None, 
                    init_q=None, eps=None):
        
        if T is None:
            T = self.q_error_table[0:self.step,:]
        if ks_vec is None:
            ks_vec = self.q_ks_vec
        if init_q is None:
            init_q = self.init_q
        if eps is None:
            eps = self.eps
        
        # x = state-action vector
        # y = list of state_actions to compute kernel_post_list
        # T = q_error_table
        # ks_vec = kernel size vector---one kernel size for each state-action dimension
        # init_dtg = value to initialize Q at
        dim = y.shape[1]
        
        if x is not None:
            xk = x * (np.abs(x) > eps)
            xk = xk / ks_vec
        yk = y / ks_vec
        Tk = T / np.concatenate([ks_vec,[1.]])    

        if x is None:
            kernel_pre = None
        else:
            if x.ndim == 1:
                kernel_pre = init_q + np.sum(Tk[:,dim] * np.exp(-np.sum( (xk-Tk[:,0:dim])**2, 1) ))
                               #init_dtg + np.sum(Tk[:,dim] * np.exp(-np.sum( (xk-Tk[:,0:dim])**2, 1) ))
            elif x.ndim == 2:
                kernel_pre = init_q + np.sum(Tk[:,dim] * np.exp(-np.sum((xk[:,np.newaxis,:] - Tk[:,0:dim])**2,-1)),1)
        kernel_post_list = init_q + np.sum(Tk[:,dim] * np.exp(-np.sum((yk[:,np.newaxis,:] - Tk[:,0:dim])**2,-1)),1)

        return [kernel_pre, kernel_post_list]
        
    def update_q(self, state_action, state_action_list, reward, done):
        
        [q_pre,q_post_list] = self.compute_q(state_action, state_action_list)
            
        q_error = self.alpha * ( (reward +  self.gamma*np.max(q_post_list)) - q_pre)
        error_list.append(q_error)
        #dtg_error = self.alpha * ((diverg + self.gamma*np.max(dtg_post_list)) - dtg_pre)
        self.q_error_table[self.step,:] = np.concatenate([state_action, [q_error]])
        self.step += 1
        
    def update_q_batch(self, sa_pre, sa_post, reward, done):
        
        # sa_pre = 2-d array (batch_size, state_action_dim)
        # sa_post = 2-d array (action_list size * batch_size, state_action_dim)
        # reward = vector (batchsize)
        # done = vector (batchsize)
        # q_error = vector (batchsize)        
        aa = self.q_error_table[:,:-1]
        nbrs = NearestNeighbors(n_neighbors=1).fit(aa)
        
        _, knnidx = nbrs.kneighbors(sa_pre)

        batchsize = sa_pre.shape[0]
        
        [q_pre,q_post1] = self.compute_q(sa_pre, sa_post)
        q_post = q_post1.reshape(batchsize,-1)
        
        q_error = self.alpha * ( (reward +  (1-done) * (self.gamma*np.max(q_post,axis=1)) - q_pre) )        
        
        self.q_error_table[knnidx,-1] += q_error[:,None] / batchsize

#%%    
class Agent(object):
    def __init__(self, transition_model, dtg, q, n_steps, rand_iters, action_list, state_dim, action_dim, div_type=0, dtg_q_flag='dtg'):
        
        self.transition_model = transition_model
        self.dtg = dtg
        self.q = q
        
        self.n_steps = n_steps
        self.rand_iters = rand_iters
        
        if action_list.ndim == 2:
            self.action_list = action_list
        elif action_list.ndim == 1:
            self.action_list = action_list[:,None]
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.div_type = div_type
    
        self.state_history = np.zeros((n_steps, state_dim*2 + action_dim + 2))
        self.diverg_history = np.zeros((n_steps, 1))
        self.diverg_list = []

        self.step = 0
        
        self.dtg_q_flag = dtg_q_flag
        
        self.dtg_sa_list = None
        
    def get_sa(self, state, action):  
        sa = np.concatenate([state,action])     
        return sa
  
    def get_sa_list(self, state):
        sa_list = np.concatenate([np.tile(state,(self.action_list.shape[0],1)),self.action_list],axis=1)    
        return sa_list
    
    def get_saem(self, state, action):
        em = self.transition_model.compute_entropy(state,action)        
        saem = np.concatenate([state,action,state,action])               
        return saem       
    
    def get_saem_list(self, state):
        em = self.transition_model.compute_entropy(state)
        ent_mean = np.concatenate([em[0][:,None], em[1]],1)
        sa_list = self.get_sa_list(state) 
        sa_list_ent_mean = np.concatenate([sa_list,sa_list],1)              
        return sa_list_ent_mean       
   
    def get_sa_batch(self, NN):
        Tlen = self.state_history[:self.step,:].shape[0]    
        
        Tinds = np.random.permutation(Tlen)[0:NN]
        
        sh = self.state_history[Tinds,:]

        sa_batch = np.concatenate([sh[:,0:self.state_dim],
                                  sh[:,self.state_dim*2:(self.state_dim*2+1)]],
                                  axis=1)
        
        sa_post = sh[:,self.state_dim:self.state_dim*2]
        
        s_list = np.kron(sa_post, np.ones((self.action_list.size,1)))
        
        a_list = np.tile(self.action_list,(sa_post.shape[0],1))
        
        sa_list_batch = np.concatenate([s_list,a_list],axis=1)
        
        reward_batch = sh[:,self.state_dim*2+self.action_dim]
        
        done_batch = sh[:,self.state_dim*2+self.action_dim+1]
        
        return [sa_batch, sa_list_batch, reward_batch, done_batch]
    
    
    def get_new_action(self, state):
        ## only perform DTG after a few random iterations        
        if self.dtg_q_flag == 'dtg':
            if self.step > self.rand_iters:
                ## get the state-action pairs we wish to evaluate for dtg
                state_action_list = self.get_saem_list(state)
            
                [_,dtg_sa_list] = self.dtg.compute_dtg(None, state_action_list)
        
                new_action = np.asarray([self.action_list[np.argmax(dtg_sa_list)]])
                
                self.dtg_sa_list = dtg_sa_list
                
            else:
                #new_action = np.asarray([np.random.choice(self.action_list)])
                new_action = np.asarray([self.action_list[np.random.randint(self.action_list.shape[0])]])
                
        elif self.dtg_q_flag == 'q':
            state_action_list = self.get_sa_list(state)
            
            [_,q_sa_list] = self.q.compute_q(None, state_action_list)
            
            new_action = np.asarray([self.action_list[np.argmax(q_sa_list)]])  
          
        return new_action
            
    def save_transition(self, state, new_state, action, reward, done):
        self.state_history[self.step] = np.concatenate([state,new_state,action,[reward],[done]])
        self.step += 1
        
    def update_transition_model(self, state, new_state, action, reward):
        self.transition_model.update_transition_model(state, new_state, action, reward)
        
    def compute_divergence(self, state, next_state, action, reward):
        #if self.step > self.rand_iters:    
        
            divs = self.transition_model.compute_divergence(state, action)
            
            diverg = divs[self.div_type]
            
            self.diverg_history[self.step-1] = diverg
       
            return diverg
            
        #else:
            
            #return None
    
    def get_dtg_input_pre(self, state, action):
        return self.get_saem(state, action)
        
    def get_dtg_input_list_post(self, state):
        return self.get_saem_list(state)
        
    def update_dtg(self, state, new_state, action, diverg):
        ## compute divergence / update dtg model
        if self.step > self.rand_iters:
            
            ## evaluate dtg for state-action (pre-transition) and list of state-actions (post-transition)
            dtg_input_pre = self.get_dtg_input_pre(state,action)
            dtg_input_list_post = self.get_dtg_input_list_post(new_state)        
            
            self.dtg.update_dtg(dtg_input_pre, dtg_input_list_post, diverg)
            
    def update_dtg_batch(self, batch_size):
        sa_batch, sa_list_batch, reward, done = self.get_sa_batch(batch_size)
            
    def update_q(self, state, new_state, action, reward, done):
        
        if self.step > self.rand_iters:
            state_action = self.get_sa(state,action)
            state_action_list = self.get_sa_list(new_state)
                        
            self.q.update_q(state_action, state_action_list, reward, done)
            
    def update_q_batch(self, batch_size):
            sa_batch, sa_list_batch, reward, done = self.get_sa_batch(batch_size)
            
            self.q.update_q_batch(sa_batch, sa_list_batch, reward, done)
            
    def plot_states(self, dim0, dim1, state_norm=None, xlims=None, ylims=None):
        # plt.figure()
        # plt.xticks([], [])
        # plt.yticks([], [])
        # plt.xlabel('location')
        # plt.ylabel('speed')
        # if state_norm is not None:
        #     plt.plot(agent.state_history[:self.step+1,dim0]*state_norm[dim0],agent.state_history[:self.step+1,dim1]*state_norm[dim1],'x')
        # else:
        #     plt.plot(agent.state_history[:self.step+1,dim0],agent.state_history[:self.step+1,dim1],'x')
        # if xlims is not None:
        #     plt.xlim(xlims)
        # if ylims is not None:
        #     plt.ylim(ylims)
        # plt.show()
        # print(self.step)
        show_x = agent.state_history[:self.step+1,dim0]*state_norm[dim0]
        show_x = ((show_x-show_x.min())/(show_x.max()-show_x.min())*10).astype(np.int32)
        
        show_y = agent.state_history[:self.step+1,dim1]*state_norm[dim1]
        show_y = ((show_y-show_y.min())/(show_y.max()-show_y.min())*11).astype(np.int32)
        
        print(len(show_y))
        running_avg_p = np.zeros([12,11])
        for i in range(len(show_x)):
            running_avg_p[show_y[i],show_x[i]]+=1
        running_avg_p=running_avg_p/(12*11)
        
        plt.figure()
        min_value = np.min(np.ma.log(running_avg_p))
        plt.imshow(np.ma.log(running_avg_p).filled(min_value), interpolation='spline16', cmap='Reds')
        #plt.imshow(running_avg_p, interpolation='spline16', cmap='Reds')

        plt.xticks([], [])
        plt.yticks([], [])
        plt.xlabel('location')
        plt.ylabel('speed')
        plt.show()
        
        
    def reset(self):
        print('resetting agent...')
        
        if self.transition_model is not None:
            self.transition_model.reset()
        if self.dtg is not None:
            self.dtg.reset()
        if self.q is not None:
            self.q.reset()
        
        self.state_history = np.zeros((self.n_steps, self.state_dim*2 + self.action_dim + 2))
        self.diverg_history = np.zeros((self.n_steps, 1))
        self.diverg_list = []
        
        self.step = 0
        self.dtg_sa_list = None

##################################################################################

#%% initialize
env = gym.make(ENVIRONMENT)

## environment constants
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
#STATE_NORM = env.observation_space.high - env.observation_space.low
STATE_NORM = np.array([1.7,.14])  # old manual normalization for mountaincar

## init models
transition_model = TransitionKernelModel(SIM_SIZE_STATE, SIM_SIZE_ACTION, 
                                         KERNEL_SIZE_STATE, KERNEL_SIZE_REWARD, ZERO_EPS, DIV_TYPE, 
                                         KNN_N, STATE_DIM, ACTION_DIM, ACTION_LIST, N_STEPS, MAX_DIVERG)
dtg = DtgKlmsModel(ALPHA, GAMMA, DTG_KERNEL_SIZE_STATE, DTG_KERNEL_SIZE_ACTION, 
                   DTG_KERNEL_SIZE_ENT, DTG_KERNEL_SIZE_MEAN, STATE_DIM, 
                   ACTION_DIM, INIT_DTG, ZERO_EPS, DTG_QUANT_EPS, N_STEPS)

q = QKlmsModel(ALPHA, GAMMA, Q_KERNEL_SIZE_STATE, Q_KERNEL_SIZE_ACTION, STATE_DIM,
                 ACTION_DIM, INIT_Q, ZERO_EPS, N_STEPS)
## init agent
agent = Agent(transition_model, dtg, q, N_STEPS, RAND_ITERS, ACTION_LIST, STATE_DIM, ACTION_DIM, DIV_TYPE, DTG_Q_FLAG)

#%% main loop
win_step_list = [0]
#def main():
## get initial state, set initial action
state = env.reset() / STATE_NORM
action = np.asarray([0])
diverg_list = []
last = 0
state_list =[]
for iii in range(N_STEPS):
    if 0:    
        env.render()
    
        
    # if np.random.rand()<0.5:      
    #     action = agent.get_new_action(state)[0]
    # else:
    #     action = [np.random.choice(ACTION_LIST)]  
    action = agent.get_new_action(state)[0]
    #action = [np.random.choice(ACTION_LIST)]  
    #print(action)
    ## take action 
    
    new_state, reward, done, info = env.step(action)
    state_list.append(new_state)
    #print(reward)
    ## normalize state    
    new_state = new_state / STATE_NORM
    


    ## if continuing after reaching goal, then reset state 
    if done and not GOAL_DONE:
        print(GOAL_DONE)
        print('done: '+str(iii-last))
        last= iii
        new_state = env.reset() / STATE_NORM
        # plot state space after reaching goal
        agent.plot_states(0,1,STATE_NORM,XLIMS,YLIMS)
        #break

        
    #     win_step_list.append(agent.step)
    
    ## record state transition 
    agent.save_transition(state, new_state, action, reward, done)
    
    ## update models
    if DTG_Q_FLAG == 'dtg':
        agent.update_transition_model(state, new_state, action, reward)
        diverg = agent.compute_divergence(state,new_state,action,reward)

        agent.update_dtg(state, new_state, action, diverg)
        diverg_list.append(diverg)
    elif DTG_Q_FLAG == 'q':
        agent.update_transition_model(state, new_state, action, reward)
        diverg = agent.compute_divergence(state,new_state,action,reward)
        reward = 0
        agent.update_q(state, new_state, action, diverg, done)
        diverg_list.append(diverg)
    #print diverg
    
    
    if UPDATE_BATCH is True:        
        if agent.step > 100:
            if agent.step % Q_REPLAY_MOD == 0:
                #print agent.step
                agent.update_q_batch(Q_REPLAY_BATCH)
    
    ## old states become new states
    state = new_state
    
    if iii % 1000 == 999:
        print(iii)
        np.save('trace_MC_KTD.npy',state_list)
        #agent.plot_states(0,1,STATE_NORM,XLIMS,YLIMS)
    
    if done and GOAL_DONE:
            print(("Episode finished after {} timesteps".format(iii+1)))
            
            break
#%%
#main()
agent.plot_states(0,1,STATE_NORM,XLIMS,YLIMS)
env.viewer.window.close()

