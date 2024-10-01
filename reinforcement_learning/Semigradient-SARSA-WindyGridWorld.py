#!/usr/bin/python
# -*- coding: utf-8 -*-
# Csar Fdez - 2024
# SARSA On policy and SemigGradient SARSA for windy grid world
import numpy as np
import copy
import matplotlib.pyplot as pl
from tqdm import tqdm

# world height
GWSizeY = 7 

# world width
GWSizeX=10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# probability for exploration
EPSILON = 0.1

# reward for each step
REWARD = -1.0

# state action pair value
Q = np.zeros((GWSizeX,GWSizeY, 4))
startState = [0, 3]
goalState = [7, 3]
Actions=['u','d','l','r']

def Environment(action,state):
    if action==0:
        state=[state[0],min(state[1]+1,GWSizeY-1)]
    if action==1:
        state=[state[0],max(state[1]-1,0)]
    if action==2:
        state=[max(state[0]-1,0),state[1]]
    if action==3:
        state=[min(state[0]+1,GWSizeX-1),state[1]]
    state=[state[0],min(state[1]+WIND[state[0]],GWSizeY-1)]
    return state

def Qgreedy(s):
    if np.random.uniform() < EPSILON:
        return np.random.randint(4)
    else:
        q=[]
        for i in range(4):
            q.append(Q[s[0],s[1],i])
        #return np.argmax(q)
        return np.random.choice([action_ for action_, value_ in enumerate(q) if value_ == np.max(q)])

NumRepetitions=50
MaxEpisodes=100
EPSILON=0.1
NumActions=len(Actions)
d=GWSizeX*GWSizeY*NumActions
w = np.zeros((GWSizeX,GWSizeY,NumActions)).reshape(d)

def X(state,a):
    ret=np.zeros((GWSizeX,GWSizeY,NumActions)) 
    ret[state[0],state[1],a]=1
    return ret.reshape(d)

def estimateQ(state,a):
    return np.dot(w,X(state,a))

def QSGgreedy(s):
    if np.random.uniform() < EPSILON:
        return np.random.randint(NumActions)
    else:
        q=[]
        for i in range(NumActions):
            q.append(estimateQ(s,i))
        return np.random.choice([action_ for action_, value_ in enumerate(q) if value_ == np.max(q)]) 


def SARSASG(alpha):
    global w
    Rarray=np.zeros((MaxEpisodes))
    for i in range(NumRepetitions):
        R=[]
        w = np.zeros((GWSizeX,GWSizeY,NumActions)).reshape(d)
        episode = 0
        while episode<MaxEpisodes:
            episodeLength=0
            currentState = startState
            currentAction = QSGgreedy(currentState)
            while currentState != goalState:
                newState =  Environment(currentAction,currentState)
                if newState == goalState:
                    w+=alpha*(-1-estimateQ(currentState,currentAction))*X(currentState,currentAction)
                    break
                else:
                    newAction = QSGgreedy(newState)
                    w+=alpha*(-1+ estimateQ(newState,newAction) -estimateQ(currentState,currentAction))*X(currentState,currentAction)

                    currentState = newState
                    currentAction = newAction
                episodeLength += 1
            R.append(episodeLength)
            episode+=1
        Rarray+=np.array((R))
    return Rarray/NumRepetitions

def SARSA(alpha):
    global Q
    Rarray=np.zeros((MaxEpisodes))
    for i in range(NumRepetitions):
        R=[]
        Q = np.zeros((GWSizeX,GWSizeY, 4))
        episode = 0
        while episode<MaxEpisodes:
            episodeLength=0
            currentState = startState
            currentAction = Qgreedy(currentState)
            while currentState != goalState:
                newState =  Environment(currentAction,currentState)

                newAction = Qgreedy(newState)

                Q[currentState[0], currentState[1], currentAction] += alpha * (REWARD + Q[newState[0], newState[1], newAction] - Q[currentState[0], currentState[1], currentAction])
                currentState = newState
                currentAction = newAction
                episodeLength += 1
            R.append(episodeLength)
            episode+=1
        Rarray+=np.array((R))
    return Rarray/NumRepetitions




def plotDifferentAlphas():
    global Q
    Alphas=[0.1,0.5,0.9]
    runsArray=np.zeros((len(Alphas),MaxEpisodes))
    runsArraySG=np.zeros((len(Alphas),MaxEpisodes))
    for i in tqdm(range(len(Alphas))):
        runsArray[i]=SARSA(Alphas[i])
        runsArraySG[i]=SARSASG(Alphas[i])

    pl.rcParams.update({'font.size': 7})
    f,ax = pl.subplots(1,1,figsize=(6,7), dpi=200, facecolor='white', edgecolor='k',sharex=True)    
    for i in range(len(Alphas)):
        color='C'+str(i)
        ax.plot(runsArray[i],color=color,linewidth=1,label="Sarsa, alpha="+str(Alphas[i]))
        ax.plot(runsArraySG[i],color=color,linewidth=1,linestyle='--',label="Sarsa-SG, alpha="+str(Alphas[i]))

    ax.set_ylabel("Episode Length")
    ax.set_xlabel("Number of episodes")
    pl.grid()
    pl.legend(loc='upper left')
    pl.show()

plotDifferentAlphas()
