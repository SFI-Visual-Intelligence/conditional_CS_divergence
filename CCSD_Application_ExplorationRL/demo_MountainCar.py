# -*- coding: utf-8 -*-
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch


def GaussianMatrix(X,Y,sigma):
    size1 = X.size()
    size2 = Y.size()
    G = (X*X).sum(-1)
    H = (Y*Y).sum(-1)
    Q = G.unsqueeze(-1).repeat(1,size1[0])
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




# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Define Q-learning function

xa = np.zeros([500000,3])
y = np.zeros([500000,2])


show_list = []
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)*\
                    np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) 
    
    # Initialize Q table
    # Q = np.random.uniform(low = -1, high = 1, 
    #                       size = (num_states[0], num_states[1], 
    #                               env.action_space.n))
    Q = np.zeros([num_states[0], num_states[1], 
                                  env.action_space.n])
    
    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []
    
    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/episodes
    step_all = 0
    
    state = env.reset()
    # Run Q learning algorithm
    for i in range(episodes):
        step = 0
        #env.render()
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        
        # Discretize state
        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
        
        xa = np.zeros([5000000,3])
        y = np.zeros([5000000,2])
        step_all = 0
        
        while done != True:   
            # Render environment for last five episodes
            if i >= (episodes - 15):
              env.render()
                
            # Determine next action - epsilon greedy strategy
            if 1:#np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]]) 
                #print(action)
            else:
                action = np.random.randint(0, env.action_space.n)
                
            # Get next state and reward
            if step > 0:
                xa[step_all] = np.concatenate((state2/np.array([1.7,.14]),np.array([action])))
                #xa[step_all] =
            state2, reward, done, info = env.step(action) 
            
            if i >= (episodes - 15):
                show_list.append(state2)
            reward_record = reward
            y[step_all] = state2/np.array([1.7,.14])
            
            if step_all>10:
                deepth = 400#400
                if step_all>deepth:
                    #print("max memory")
                    Tlen = deepth
                    x1 = xa[step_all-deepth:step_all+1][:Tlen//2]

                    y_old =  y[step_all-deepth:step_all+1][:Tlen//2]
        
                
                    x2 = xa[step_all-deepth:step_all+1][Tlen//2:Tlen//2*2]
                    y_new = y[step_all-deepth:step_all+1][Tlen//2:Tlen//2*2]
        
                
        
                
                    csd_our = CSD_4(x1,x2,y_old,y_new,sigma = 0.1)
                    if reward >0:
                        print([reward,csd_our])
                    reward = csd_our-1#reward#csd_our#reward + 0.01*csd_our
                    # if step_all%1000 ==999:
                    #       print([reward,step_all,Tlen])
                    
                    
                else:
                    Tlen = step_all
                    x1 = xa[:Tlen//2]
                    y_old =  y[:Tlen//2]
        
                
                    x2 = xa[Tlen//2:Tlen//2*2]
                    y_new = y[Tlen//2:Tlen//2*2]
        
                
        
                
                    csd_our = CSD_4(x1,x2,y_old,y_new,sigma = 0.1)#0.1
                    
                    reward = csd_our-1#reward#csd_our#reward + 0.01*csd_our
                    # if step_all%100 ==0:
                    #     print([csd_our,step_all])
                
            else:
                reward = -1#-1#reward#-1#reward
                
            step += 1
            step_all+=1
            #print(step)
            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            #Allow for terminal states
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
                
            # Adjust Q value for current state
            else:
                delta = learning*(reward + 
                                 discount*np.max(Q[state2_adj[0], 
                                                   state2_adj[1]]) - 
                                 Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta
                                     
            # Update variables
            tot_reward += reward
            state_adj = state2_adj
        #print(step)
        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction
        
        # Track rewards
        reward_list.append(tot_reward)
        
        if (i+1) % 1 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            
        if (i+1) % 1 == 0:    
            print('Episode {} Average Reward: {} Step: {}'.format(i+1, ave_reward, step_all))
            
    env.close()
    
    return ave_reward_list, show_list

# Run Q-learning algorithm
import time
start = time.time()
rewards, show_list_1 = QLearning(env, 0.2, 0.9, 0.8, 0, 100)
end = time.time()


