#!/usr/bin/python
# -*- coding: utf-8 -*-
# Csar Fdez - 2024
# Q-learning for Cart Pole (GYM) environment
import numpy as np
import gym
import argparse
import matplotlib.pyplot as pl
import pickle
from tqdm import tqdm

def InitDiscretizer(MinV,MaxV,N):
    #MinV:  M-dimensional vector with the minimums of each feature
    #MaxV:  M-dimensional vector with the maximums of each feature
    #N: Number of divisions per feature
    NumFeat=len(MinV)
    delta=[]
    thresh=[]
    for i in range(NumFeat):
        delta.append((MaxV[i]-MinV[i])/N)
        temp=[]
        for j in range(N-1):
            temp.append(MinV[i]+(j+1)*delta[i])
        thresh.append(temp)
    return thresh


def Discretize(data):
    #data:  NumFeat-dimensional vector with the observation
    ret=np.zeros([NumFeat],dtype=int)
    for i in range(NumFeat):
        for j in range(N-1):
            if data[i]>Thresh[i][j]:
                ret[i]=j+1
            else:
                break
    return ret


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

NumFeat=4
N=64
# Observed position, speed, angle and angular speed uses to be
# inside margins (-1,1), (-2,2), (-1,1) and (-2,2) respectively
Thresh=InitDiscretizer([-1,-2,-1,-2],[1,2,1,2],N)

EPSILON = 0.1
episodeLimit=3000
Q = np.zeros((N,N,N,N, num_actions))
def Qgreedy(s):
    if np.random.uniform() < EPSILON:
        return np.random.randint(2)
    else:
        q=[]
        for i in range(num_actions):
            q.append(Q[s[0],s[1],s[2],s[3],i])
        return np.argmax(q)


def oneEpisodeQL(episode,ALPHA):
    # track the total time steps in this episode
    global Q
    episodeLength = 0
    currentState = Discretize(env.reset()[0])
    currentAction = Qgreedy(currentState)
    done=False
    while not done:
        newStatecont,rew,done,_,_ =  env.step(currentAction)
        newState=Discretize(newStatecont)

        Q[currentState[0], currentState[1],currentState[2], currentState[3], currentAction] += ALPHA * (rew + np.max(Q[newState[0], newState[1],newState[2], newState[3],:]) - Q[currentState[0], currentState[1],currentState[2], currentState[3], currentAction])
        currentState = newState
        currentAction = Qgreedy(currentState)
        episodeLength += 1
        if episodeLength>episodeLimit:
            done=True
    return episodeLength

def trajectory():
    global EPSILON
    episodeLength = 0
    EPSILON=0
    currentState = Discretize(env.reset()[0])
    currentAction = Qgreedy(currentState)
    done=False
    counter=0
    while not done:
        print(counter)
        newStatecont,rew,done,_,_ =  env.step(currentAction)
        newState=Discretize(newStatecont)
        if args.render:
            env.render()
        currentState = newState
        currentAction = Qgreedy(currentState)
        episodeLength += 1
        if episodeLength>episodeLimit:
            done=True
        counter+=1
    return episodeLength

alpha=0.5
NumEpTrain=50000
convolveWindow=500   # window to run average mean
episodesLength=[]

if args.train:
    for ep in tqdm(range(NumEpTrain)):
        episodesLength.append(oneEpisodeQL(ep,alpha))
    filehandler = open("Q.pk", 'wb') 
    pickle.dump(Q, filehandler)
    filehandler.close()
    filehandler = open("QepisodesLength.pk", 'wb') 
    pickle.dump(episodesLength, filehandler)
    filehandler.close()

elif args.render:
    env = gym.make(args.environment, render_mode="human")
    filehandler = open("Q.pk", 'rb') 
    Q = pickle.load(filehandler)
    trajectory()
 
else:
    filehandler = open("QepisodesLength.pk", 'rb') 
    el = pickle.load(filehandler)
    elfiltered = np.convolve(np.array((el)), np.ones(convolveWindow)/convolveWindow, mode='valid')
    pl.rcParams.update({'font.size': 7})
    f,ax = pl.subplots(1,1,figsize=(6,7), dpi=200, facecolor='white', edgecolor='k',sharex=True)    
    #ax.plot(    el,linewidth=1,label="QL CartPole")
    ax.plot(    elfiltered,linewidth=2,label="QL CartPole filtered")

    ax.set_ylabel("Episode Length")
    ax.set_xlabel("Number of training episodes")
    pl.grid()
    pl.legend(loc='upper left')
    pl.show()

