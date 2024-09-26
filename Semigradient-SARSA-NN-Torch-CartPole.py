#!/usr/bin/python
# -*- coding: utf-8 -*-
# Csar Fdez - 2024
# SemiGradient SARSA for Cart Pole (GYM) environment with Neural Network
import numpy as np
import gym
import argparse
import matplotlib.pyplot as pl
import pickle
from tqdm import tqdm
import torch
from torch import optim
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='render the environment')
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

learning_rate=1e-3

model = torch.nn.Sequential(
    torch.nn.Linear(input_dimension, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, num_actions)
)
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

EPSILON = 0.5
gamma=1
episodeLimit=1000
def Qgreedy(s):
    if np.random.uniform() < EPSILON:
        return np.random.randint(num_actions)
    else:
        x=torch.tensor(s).float()
        return torch.argmax(model(x)).item() 

def estimateQ(s,a):
    x=torch.tensor(s).float()
    return model(x)[a]        


def oneEpisodeQL():
    # track the total time steps in this episode
    episodeLength = 0
    currentState = env.reset()[0]
    currentAction = Qgreedy(currentState)
    done=False
    while not done:
        newState,rew,done,_,_ =  env.step(currentAction)
        if done:
            y=estimateQ(currentState,currentAction)
            loss=loss_fn(y,torch.tensor(rew).float())
            model.zero_grad()
            loss.backward()  # Computes the gradient
            optimizer.step()   # Optimizes weights
            break            
        else:
            newAction = Qgreedy(newState)
            yy=gamma*estimateQ(newState,newAction)
            y=estimateQ(currentState,currentAction)
            loss=loss_fn(y-yy,torch.tensor(rew).float())
            model.zero_grad()
            loss.backward()
            optimizer.step()

        currentState = newState
        currentAction = newAction
        episodeLength += 1
        if episodeLength>episodeLimit:
            done=True
    return episodeLength

def trajectory():
    # track the total time steps in this episode
    global EPSILON
    episodeLength = 0
    EPSILON=0
    currentState = env.reset()[0]
    currentAction = Qgreedy(currentState)
    done=False
    counter=0
    while not done:
        print(counter)
        newState,rew,done,_,_ =  env.step(currentAction)
        if args.render:
            env.render()
        currentState = newState
        currentAction = Qgreedy(currentState)
        episodeLength += 1
        if episodeLength>episodeLimit:
            done=True
        counter+=1
    return episodeLength

NumEpTrain=10000
convolveWindow=100   # window to run average mean
episodesLength=[]

modelname="SGSARSANNtorchmodel"
dataname="SGSARSANNtorchepisodesLength.pk"


if args.train:
    epinit=0
    if os.path.exists(modelname) and os.path.exists(dataname): 
        model.load_state_dict(torch.load(modelname))
        filehandler = open(dataname, 'rb') 
        episodesLength = pickle.load(filehandler)
        epinit=len(episodesLength)

    for ep in tqdm(range(epinit,epinit+NumEpTrain)):
        l=oneEpisodeQL()
        episodesLength.append(l)
        if l>=episodeLimit:
            torch.save(model.state_dict(),modelname)   

        EPSILON=0.5/(ep/1000+1)
    filehandler = open(dataname, 'wb') 
    pickle.dump(episodesLength, filehandler)
    filehandler.close()

elif args.render:
    env = gym.make(args.environment, render_mode="human")
    model.load_state_dict(torch.load(modelname))
    trajectory()
 
else:
    filehandler = open(dataname, 'rb') 
    el = pickle.load(filehandler)
    elfiltered = np.convolve(np.array((el)), np.ones(convolveWindow)/convolveWindow, mode='valid')
    filehandler = open("QepisodesLength.pk", 'rb') 
    el = pickle.load(filehandler)
    elfiltered2 = np.convolve(np.array((el)), np.ones(convolveWindow)/convolveWindow, mode='valid')
    
    pl.rcParams.update({'font.size': 7})
    f,ax = pl.subplots(1,1,figsize=(6,7), dpi=200, facecolor='white', edgecolor='k',sharex=True)    
    ax.plot(    elfiltered,linewidth=2,label="Semi-Gradient SARSA NN CartPole")
    ax.plot(    elfiltered2,linewidth=2,label="QL CartPole")

    ax.set_ylabel("Episode Length (averaged over last "+str(convolveWindow)+" episodes)")
    ax.set_xlabel("Number of training episodes")
    pl.grid()
    pl.legend(loc='upper right')
    pl.show()

