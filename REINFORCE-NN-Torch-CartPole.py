#!/usr/bin/python
# -*- coding: utf-8 -*-
# Csar Fdez - 2024
# REINFORCE for Cart Pole (GYM) environment with Neural Network
# Based on  https://www.datahubbs.com/reinforce-with-pytorch/
# See https://github.com/hagerrady13/Reinforce-PyTorch for Baseline version
import numpy as np
import gym
import argparse
import matplotlib.pyplot as pl
import pickle
from tqdm import tqdm
import os
import torch
from torch import nn
from torch import optim

parser = argparse.ArgumentParser()
parser.add_argument("--environment", "-e", default = "CartPole-v1")
parser.add_argument("--train", action = 'store_true')
args = parser.parse_args()
env = gym.make(args.environment)

input_dimension=int(env.observation_space.shape[0])
num_actions=int(env.action_space.n)

print("Env space:", input_dimension)
print(env.observation_space.high)
print(env.observation_space.low)

print("Num Actions:", num_actions)

gamma=1
episodeLimit=1000
learning_rate=1e-3




def Model():
    m=nn.Sequential( nn.Linear(input_dimension, 32), nn.ReLU(), nn.Linear(32, num_actions), nn.Softmax(dim=-1))
    return m


def predict(state):
    action_probs = model(torch.FloatTensor(state))
    return action_probs


model=Model()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def discount_rewards(rewards):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    #return r - r.mean()   # Baseline constant = mean Gains
    return r             # No baseline

action_space = np.arange(num_actions)

batch_size=2

def oneBatchREINFORCE():
    global episodesLength
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 0

    while batch_counter<batch_size:
        state=env.reset()[0]
        done=False          
        states = []
        rewards = []
        actions = []
        while not done:
            action_probs = predict(state).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            next_state, reward, done, _ , _=env.step(action)
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            state=next_state

            if done or sum(rewards)>episodeLimit:
                done=True
                batch_rewards.extend(discount_rewards(rewards))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                episodesLength.append(sum(rewards))
                if batch_counter==(batch_size-1):
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(np.array(batch_states))
                    reward_tensor =torch.FloatTensor(batch_rewards) 
                    action_tensor = torch.LongTensor(batch_actions)
                    # Compute loss
                    logprob = torch.log(predict(state_tensor))
                    selected_logprobs = reward_tensor * logprob[np.arange(len(action_tensor)), action_tensor]
                    loss = -selected_logprobs.mean()
                    loss.backward() # Computes the gradient
                    optimizer.step() # Optimizes weights


NumEpTrain=5000
episodesLength=[]
convolveWindow=100

modelname="Reinforcetorch"
dataname="ReinforcetorchepisodesLength.pk"

if args.train:
    epinit=0
    for ep in tqdm(range(epinit,epinit+NumEpTrain)):
        oneBatchREINFORCE()

    filehandler = open(dataname, 'wb') 
    pickle.dump(episodesLength, filehandler)
    filehandler.close()
    torch.save(model.state_dict(),modelname)
else:
    filehandler = open(dataname, 'rb') 
    el = pickle.load(filehandler)
    elfiltered = np.convolve(np.array((el)), np.ones(convolveWindow)/convolveWindow, mode='valid')

    filehandler = open("SGSARSANNtorchepisodesLength.pk", 'rb') 
    el2 = pickle.load(filehandler)
    elfiltered2 = np.convolve(np.array((el2)), np.ones(convolveWindow)/convolveWindow, mode='valid')


    pl.rcParams.update({'font.size': 7})
    f,ax = pl.subplots(1,1,figsize=(6,7), dpi=200, facecolor='white', edgecolor='k',sharex=True)    
    ax.plot(    elfiltered,linewidth=2,label="REINFORCE Torch CartPole")
    ax.plot(    elfiltered2,linewidth=2,label="SG-SARSA-Torch CartPole")

    ax.set_ylabel("Episode Length (averaged over last "+str(convolveWindow)+" episodes)")
    ax.set_xlabel("Number of training episodes")
    pl.grid()
    pl.legend(loc='upper right')
    pl.show()

