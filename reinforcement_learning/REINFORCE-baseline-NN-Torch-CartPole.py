#!/usr/bin/python
# -*- coding: utf-8 -*-
# Csar Fdez - 2024
# REINFORCE with baseline for Cart Pole (GYM) environment with Neural Network
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
learning_rate_policy=1e-3
learning_rate_value=1e-3


def ModelPolicy():
    m=nn.Sequential( nn.Linear(input_dimension, 32), nn.ReLU(), nn.Linear(32, num_actions), nn.Softmax(dim=-1))
    return m

def ModelValue():
    m=nn.Sequential( nn.Linear(input_dimension, 32), nn.ReLU(), nn.Linear(32, 1), nn.ReLU())
    return m

def predictPolicy(state):
    action_probs = modelPolicy(torch.FloatTensor(state))
    return action_probs

def predictValue(state):
    value = modelValue(torch.FloatTensor(state))
    return value

modelPolicy=ModelPolicy()
modelValue=ModelValue()
optimizerPolicy = optim.Adam(modelPolicy.parameters(), lr=learning_rate_policy)
optimizerValue = optim.Adam(modelValue.parameters(), lr=learning_rate_value)

def discount_rewards(rewards,values):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return r-values 

action_space = np.arange(num_actions)

batch_size=2

def oneBatchREINFORCE():
    global episodesLength
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_values= []
    batch_counter = 0

    while batch_counter<batch_size:
        state=env.reset()[0]
        done=False          
        states = []
        rewards = []
        actions = []
        values = []
        while not done:
            action_probs = predictPolicy(state).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            value = predictValue(state).detach().item()
            next_state, reward, done, _ , _=env.step(action)
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            values.append(value)
            state=next_state

            if done or sum(rewards)>episodeLimit:
                done=True
                batch_rewards.extend(discount_rewards(rewards,values))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_values.extend(values)
                batch_counter += 1
                episodesLength.append(sum(rewards))
                if batch_counter==(batch_size-1):
                    optimizerPolicy.zero_grad()
                    optimizerValue.zero_grad()
                    state_tensor = torch.FloatTensor(np.array(batch_states))
                    reward_tensor =torch.FloatTensor(batch_rewards) 
                    action_tensor = torch.LongTensor(batch_actions)
                    # Compute loss
                    logprob = torch.log(predictPolicy(state_tensor))
                    selected_logprobs = reward_tensor * logprob[np.arange(len(action_tensor)), action_tensor]
                    loss = -selected_logprobs.mean()

                    val = predictValue(state_tensor)
                    lossV =(reward_tensor*val-0.5*val.pow(2)).mean()

                    loss.backward()
                    lossV.backward()
                    optimizerPolicy.step()
                    optimizerValue.step()


NumEpTrain=5000
episodesLength=[]
convolveWindow=100

modelname="Reinforcebaselinetorch"
dataname="ReinforcebaselinetorchepisodesLength.pk"

if args.train:
    epinit=0

    for ep in tqdm(range(epinit,epinit+NumEpTrain)):
        oneBatchREINFORCE()

    filehandler = open(dataname, 'wb') 
    pickle.dump(episodesLength, filehandler)
    filehandler.close()
    torch.save(modelPolicy.state_dict(),modelname)
else:
    filehandler = open(dataname, 'rb') 
    el = pickle.load(filehandler)
    elfiltered = np.convolve(np.array((el)), np.ones(convolveWindow)/convolveWindow, mode='valid')

    filehandler = open("ReinforcetorchepisodesLength.pk", 'rb') 
    el2 = pickle.load(filehandler)
    elfiltered2 = np.convolve(np.array((el2)), np.ones(convolveWindow)/convolveWindow, mode='valid')


    pl.rcParams.update({'font.size': 7})
    f,ax = pl.subplots(1,1,figsize=(6,7), dpi=200, facecolor='white', edgecolor='k',sharex=True)    
    ax.plot(    elfiltered,linewidth=2,label="REINFORCE baseline  CartPole")
    ax.plot(    elfiltered2,linewidth=2,label="REINFORCE  CartPole")

    ax.set_ylabel("Episode Length (averaged over last "+str(convolveWindow)+" episodes)")
    ax.set_xlabel("Number of training episodes")
    pl.grid()
    pl.legend(loc='lower right')
    pl.show()

