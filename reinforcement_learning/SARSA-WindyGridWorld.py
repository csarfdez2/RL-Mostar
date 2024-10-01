#!/usr/bin/python
# -*- coding: utf-8 -*-
# Csar Fdez - 2024
# SARSA On policy for windy grid world
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
        return np.argmax(q)

# play for an episode
def oneEpisode(ALPHA):
    # track the total time steps in this episode
    episodeLength = 0
    currentState = startState
    currentAction = Qgreedy(currentState)
    while currentState != goalState:
        newState =  Environment(currentAction,currentState)

        newAction = Qgreedy(newState)

        Q[currentState[0], currentState[1], currentAction] += ALPHA * (REWARD + Q[newState[0], newState[1], newAction] - Q[currentState[0], currentState[1], currentAction])
        currentState = newState
        currentAction = newAction
        episodeLength += 1
    return episodeLength-1

def createTrajectory():
    global EPSILON
    EPSILON=0
    trajectory=[]
    currentState = startState
    currentAction = Qgreedy(currentState)
    trajectory.append(currentState)
    while currentState != goalState:
        newState =  Environment(currentAction,currentState)
        newAction = Qgreedy(newState)
        currentState = newState
        currentAction = newAction
        trajectory.append(currentState)

    return trajectory

episodeLimit = 500

def plotDifferentAlphas():
    global Q
    Alphas=[0.1,0.5,0.9]
    numRepetitions=50
    runsArray=np.zeros((len(Alphas),episodeLimit))
    for i in tqdm(range(numRepetitions)):
        runs=[]
        for a in Alphas:
            episodesLength=[]
            for ep in range(episodeLimit):
                episodesLength.append(oneEpisode(a))
            runs.append(episodesLength)
            Q = np.zeros((GWSizeX,GWSizeY, 4))
            runsArray[Alphas.index(a)]+=np.array(runs[Alphas.index(a)])
    runsArray=runsArray/numRepetitions

    pl.rcParams.update({'font.size': 7})
    f,ax = pl.subplots(1,1,figsize=(6,7), dpi=200, facecolor='white', edgecolor='k',sharex=True)    
    for i in range(len(Alphas)):
        ax.plot(runsArray[i],linewidth=1,label="Sarsa, alpha="+str(Alphas[i]))

    ax.set_ylabel("Episode Length")
    ax.set_xlabel("Number of episodes")
    pl.grid()
    pl.legend(loc='upper left')
    pl.show()


plotDifferentAlphas()

# Now plot Optimal Policy for alpha=0.5
for ep in range(episodeLimit):
    oneEpisode(0.5)


pl.rcParams.update({'font.size': 7})
f,ax = pl.subplots(1,1,figsize=(6,7), dpi=200, facecolor='white', edgecolor='k',sharex=True)    

# Plot a trajectory example

traject=createTrajectory()
traj=np.array(traject)
trajx=traj[:,0]+0.5
trajy=traj[:,1]+0.5
for i in range(GWSizeX):
    for j in range(GWSizeY):
        bestact=np.argmax(Q[i,j,:])
        x=i
        y=j
        dx=0
        dy=0
        if bestact==0:
            dy+=0.25
        elif bestact==1:
            dy-=0.25
        elif bestact==2:
            dx-=0.25
        else:
            dx+=0.25
        ax.arrow(x+0.5,y+0.5,dx,dy,head_width=0.1)

ax.text(startState[0]+0.25,startState[1]+0.25,"S",color='red',fontsize=16)
ax.text(goalState[0]+0.25,goalState[1]+0.25,"E",color='green',fontsize=16)
ax.plot(trajx,trajy)
ax.text(GWSizeX/2,GWSizeY/2-0.4,"Length: "+str(len(trajx)-1),fontsize=10,color='blue')
ax.set_xlim(0,GWSizeX)
ax.set_ylim(0,GWSizeY)
ax.set_xticks(list(range(GWSizeX)))
ax.set_yticks(list(range(GWSizeY)))
pl.grid()
pl.show()
