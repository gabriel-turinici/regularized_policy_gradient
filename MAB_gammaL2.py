# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:22:04 2024

@author: S. Anita & G. Turinici

some parts (first class) is taken from https://github.com/kamenbliznashki/sutton_barto
(as it was in March 2024)
The MAB update + regulatization part is re-written.

"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#import warnings

def save_fig(name, ch = ''):
#    plt.savefig('figures_{}_{}.pdf'.format(ch, name))#if error reload shell
    plt.savefig('figures_{}_{}.jpg'.format(ch, name))
    plt.close()



def softmax(x):
    x=x-np.max(x)
    return np.exp(x)/np.sum(np.exp(x))



# --------------------
# Bandit problem and functions
# --------------------

class BaseBandit:
    """
    Base class implementation of Section 2.3: The 10-armed Testbed
    """
    def __init__(
            self,
            k_arm=10,     # number of arms
            eps=0,        # explore with prob eps; exploit with prob 1-eps
            initial_q=0,  # initial action-value estimates
            true_q_mean=0, # true reward mean
            init_h_val=None#initial H
            ):
        self.k_arm = k_arm
        self.possible_actions = np.arange(self.k_arm)
        self.eps = eps
        self.initial_q = initial_q
        self.true_q_mean = true_q_mean
        self.init_h_val=init_h_val
        self.reset()

    def reset(self):
        #for each bandit the action values q_*(a) were selected from a normal distribution with 0 mean and variance 1
        self.q_true = np.random.randn(self.k_arm) + self.true_q_mean

        # initialize 'A simple bandit algorithm' p24
        self.q_estimate = np.zeros(self.k_arm) + self.initial_q
        self.action_count = np.zeros(self.k_arm)

        # initialize h for gradient alg
        #self.h_vals = np.zeros(self.k_arm)
        if (self.init_h_val is None):
            self.h_vals=np.zeros(self.k_arm)
        else:
            self.h_vals=self.init_h_val.copy()

        #for a biased initial distribution
        #self.h_vals[0] = 5
#        self.softmax = np.zeros(self.k_arm)
        self.softmax = softmax(self.h_vals)

        # record how often the optimal action is selected
        self.optimal_action_freq = 0


    def act(self):
        # explore with prob eps; exploit with prob 1-eps
        if np.random.rand() < self.eps: # explore
            action = np.random.choice(self.possible_actions)
        else:
            action = np.argmax(self.q_estimate)
        return action

    def reward(self, action_idx):
        #the actual reward is selected from a normal distribution with mean q_*(A_t) and variance 1
        return np.random.randn() + self.q_true[action_idx]

    def update_q(self, action, reward):
        # simple average update
        self.q_estimate[action] += 1/self.action_count[action] * (reward - self.q_estimate[action])

    def step(self):
        # single loop in 'A simple bandit algorithm'
        action = self.act()
        reward = self.reward(action)
        self.action_count[action] += 1
        self.update_q(action, reward)

        #update of average for optimal action frequency
        if action == np.argmax(self.q_true):  # action == best possible action
            self.optimal_action_freq += 1/np.sum(self.action_count) * (1 - self.optimal_action_freq)

        return action, reward


class GradientBandit_gamma(BaseBandit):
    """
    Implementation of Section 2.8 Gradient Bandit Algorithm
    """
    def __init__(
            self,
            baseline=True,  # use average returns as a baseline for gradient calculation
#            step_size=0.1,  # exponential weighted avg param
            eps=0,
            gamma0=0,gamma_decay_cst=0.0,rho_decay_cst=0.0,rho0=0.1,init_h_val=None,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.baseline = baseline
#        self.step_size = step_size
        self.average_reward = 0
        self.eps =eps
        self.gamma0 =gamma0
        self.rho0 =rho0
        self.gamma_decay_cst =gamma_decay_cst
        self.rho_decay_cst =rho_decay_cst
        self.nt =0.0 #counter for time : nt=0 is initial value, updated at each use of "update_q" function
        self.rho_t =rho0 
        self.gamma_t =gamma0 
        self.init_h_val=init_h_val
            
        


    def act(self):
        if np.random.rand() < self.eps: # explore
            return np.random.choice(self.possible_actions)
        else:            
            #e = np.exp(self.h_vals-self.h_vals[np.argmax(self.h_vals)])
            #e = self.h_vals-np.max(self.h_vals)
            self.softmax = softmax(self.h_vals)
            #if np.isnan(1/np.nansum(e))==True: 
            #    e = e / self.k_arm
            #self.softmax = e / np.sum(e)
            return np.random.choice(self.possible_actions, p=self.softmax)

    def update_q(self, action, reward):
        # avg rewards serve as a baseline; if reward > baseline then prob(action) is increased
        # first do online update of average reward
        # (note n number of steps == sum of action counts since at each step only one action is chosen
        #baseline = self.average_reward if self.baseline else 0
        # gradient update:
        mask = np.zeros_like(self.softmax)
        mask[action] = 1
        self.rho_t =self.rho0/(1+self.nt*self.rho_decay_cst) 
        self.gamma_t =self.gamma0/(1+self.nt*self.gamma_decay_cst) 
        #need to clip values that are too large and risk to overflow; these are all pathological cases
        self.h_vals=np.clip(self.h_vals,-1.0e+200,1.0e+200) 
        self.h_vals += self.rho_t * ((reward - self.average_reward) * (mask - self.softmax)-self.gamma_t*self.h_vals)    
        self.q_estimate[action] += 1/self.action_count[action] * (reward - self.q_estimate[action])
        self.nt +=1.0


# --------------------
# Evaluate a list of bandit problems
# --------------------

def run_bandits(bandits, n_runs, n_steps):
    """ simulates a list of bandit running each for n_teps and then averaging over n_runs """

    rewards = np.zeros((len(bandits), n_runs, n_steps))
    optimal_action_freqs = np.zeros_like(rewards)

    for b, bandit in enumerate(bandits):
        for run in tqdm(range(n_runs)):
            # runs are independent; so reset bandit
            bandit.reset()
            bandit.nt =0.0 #counter for time : nt=0 is initial value, updated at each use of "update_q" function
            bandit.rho_t =bandit.rho0 
            bandit.gamma_t =bandit.gamma0 

            for step in range(n_steps):
                bandit.g_step=step #erase after test
                # step bandit (act -> reward)
                action, reward = bandit.step()
                # record reward averages and optimal action frequence
                rewards[b, run, step] = reward/np.amax(bandit.q_true)
                if action == np.argmax(bandit.q_true):
                    optimal_action_freqs[b, run, step] = 1
        print(bandit.q_true)    
        print(bandit.q_estimate)
        print(bandit.action_count)
        print(bandit.h_vals)

    # average across the n_runs
    avg_rewards = rewards.mean(axis=1)
    avg_optimal_action_freqs = optimal_action_freqs.mean(axis=1)

    return avg_rewards, avg_optimal_action_freqs



# --------------------
# Figure 2.5: Average performance of the gradient bandit algorithm with and without a reward baseline
# on the 10-armed testbed when the q⇤(a) are chosen to be near +4 rather than near zero.
# --------------------

def fig_2_5(runs=2000, steps=1000,filename="fig_2_5_reward",gamma_list=[0.0,0.01,10.0],
        gamma_decay_cst=0.0,rho_decay_cst=0.0,rho0=0.1,init_h_val=None):
    bandits = [GradientBandit_gamma(true_q_mean=4, 
                                    gamma0=gamma_val,gamma_decay_cst=gamma_decay_cst,
                                    rho_decay_cst=rho_decay_cst,rho0=rho0,init_h_val=init_h_val)
               for gamma_val in gamma_list]
    fig_2_5.bandits=bandits
    avg_rewards, avg_optimal_action_freqs = run_bandits(bandits, runs, steps)

    # plot results rewards
    plt.figure('fig_2_5_reward')
    for i, bandit in enumerate(bandits):
        if(bandit.gamma_decay_cst==0.0):
            plt.scatter(list(range(len(avg_rewards[i]))),avg_rewards[i],
#            plt.plot(avg_rewards[i],
                    #label='$\gamma_0={}$'.format(bandit.gamma0))
                    label=r'$\gamma={}$'.format(bandit.gamma0))
        else:
            plt.scatter(list(range(len(avg_rewards[i]))),avg_rewards[i],
#            plt.plot(avg_rewards[i],
                    label=r'$\gamma_0={}$'.format(bandit.gamma0))

    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    save_fig(filename)

if __name__ == '__main__':
    no_arms=10
#    warnings.filterwarnings("error")
    H_uniform=np.zeros(no_arms)
    fig_2_5(runs=1000,steps=2000,filename="nonbiased_gamma_0and0_01and10",
            gamma_list=[0.0,0.01,10.0],gamma_decay_cst=0.0,
            rho_decay_cst=0.0,rho0=0.05,init_h_val=H_uniform)
    print('figure 1 left, done')

    H_biased=np.zeros(no_arms)
    H_biased[0]=5.0

    fig_2_5(runs=1000,steps=2000,filename="biased_gamma_0and0_01",
            gamma_list=[0.0,0.01],gamma_decay_cst=0.0,
            rho_decay_cst=0.0,rho0=0.05,init_h_val=H_biased)
    print('figure 1 right, done')

    fig_2_5(runs=1000,steps=2000,filename="biased_gamma_0and0_01and10_rhot",
                gamma_list=[0.0,0.01,10.0],gamma_decay_cst=0.0,
                rho_decay_cst=0.05,rho0=1.0,init_h_val=H_biased)
    print('figure 2, done')

    fig_2_5(runs=1000,steps=2000,filename="biased_gamma_0and10_gammat_rhot",
            gamma_list=[0.0,10.0],gamma_decay_cst=0.2,
            rho_decay_cst=0.05,rho0=1.0,init_h_val=H_biased)
    print('figure 3, done')

