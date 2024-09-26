#!/usr/bin/python
# -*- coding: utf-8 -*-
# Csar Fdez - 2024
# SARSA and Q-learning on windy grid world
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

# Used for SARSA
def Qgreedy(s):
    if np.random.uniform() < EPSILON:
        return np.random.randint(4)
    else:
        q=[]
        for i in range(4):
            q.append(Q[s[0],s[1],i])
        return np.argmax(q)

# Used for QLearning
def Qopt(s):
    q=[]
    for i in range(4):
        q.append(Q[s[0],s[1],i])
    return np.argmax(q)

# play for an episode
def oneEpisodeQL(episode,ALPHA):
    # track the total time steps in this episode
    episodeLength = 0
    currentState = startState
    currentAction = Qgreedy(currentState)
    while currentState != goalState:
        newState =  Environment(currentAction,currentState)

        Q[currentState[0], currentState[1], currentAction] += ALPHA * (REWARD + np.max(Q[newState[0], newState[1],:]) - Q[currentState[0], currentState[1], currentAction])
        currentState = newState
        currentAction = Qopt(currentState)
        episodeLength += 1
    return episodeLength-1

def oneEpisodeSARSA(episode,ALPHA):
    # track the total time steps in this episode
    episodeLength = 0
    currentState = startState
    currentAction = Qgreedy(currentState)
    while currentState != goalState:
        newState =  Environment(currentAction,currentState)

        newAction = Qgreedy(newState)

        Q[currentState[0], currentState[1], currentAction] += ALPHA * (REWARD + Q[newState[0], newState[1],newAction] - Q[currentState[0], currentState[1], currentAction])
        currentState = newState
        currentAction = newAction
        episodeLength += 1
    return episodeLength-1


Alphas=[0.5]

episodeLimit = 500
numRepetitions=50
runsArrayQL=np.zeros((len(Alphas),episodeLimit))
runsArraySARSA=np.zeros((len(Alphas),episodeLimit))
    
for i in tqdm(range(numRepetitions)):
    runsQL=[]
    runsSARSA=[]
    for a in Alphas:
        episodesLengthQL=[]
        episodesLengthSARSA=[]
        for ep in range(episodeLimit):
            episodesLengthQL.append(oneEpisodeQL(ep,a))
        runsQL.append(episodesLengthQL)
        runsArrayQL[Alphas.index(a)]+=np.array(runsQL[Alphas.index(a)])
        Q = np.zeros((GWSizeX,GWSizeY, 4))

        for ep in range(episodeLimit):
            episodesLengthSARSA.append(oneEpisodeSARSA(ep,a))
        runsSARSA.append(episodesLengthSARSA)
        runsArraySARSA[Alphas.index(a)]+=np.array(runsSARSA[Alphas.index(a)])
        Q = np.zeros((GWSizeX,GWSizeY, 4))


runsArrayQL=runsArrayQL/numRepetitions
runsArraySARSA=runsArraySARSA/numRepetitions

pl.rcParams.update({'font.size': 7})
f,ax = pl.subplots(1,1,figsize=(6,7), dpi=200, facecolor='white', edgecolor='k',sharex=True)    
for i in range(len(Alphas)):
    ax.plot(runsArrayQL[i],linewidth=1,label="QL, alpha="+str(Alphas[i]))
    ax.plot(runsArraySARSA[i],linewidth=1,label="SARSA, alpha="+str(Alphas[i]))

ax.set_ylabel("Episode Length")
ax.set_xlabel("Number of episodes")
pl.grid()
pl.legend(loc='upper left')
pl.show()


exit(0)
# Compute 100 trajectories and average
def trajectory(q):
    global EPSILON,Q
    Q=q
    EPSILON=0
    count=0
    currentState = startState
    currentAction = Qgreedy(currentState)
    while currentState != goalState:
        newState =  Environment(currentAction,currentState)
        newAction = Qgreedy(newState)
        currentState = newState
        currentAction = newAction
        count+=1
    return count

trajQL=[]
trajSARSA=[]
for i in range(1000):
    trajQL.append(trajectory(QQL))
    trajSARSA.append(trajectory(QSARSA))


print("SARSA average: ",np.average(np.array(trajSARSA)))
print("QL average: ",np.average(np.array(trajQL)))

print(QQL[0,0,:])
print(QSARSA[0,0,:])
