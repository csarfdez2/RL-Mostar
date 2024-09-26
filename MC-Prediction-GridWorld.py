#!/usr/bin/python
# -*- coding: utf-8 -*-
# Csar Fdez - 2020
#  Algo First Visit. Page 76
import numpy as np
import copy

def Environment(action,state):
    if state==[0,1]:
        state=[4,1]
        return state,10
    if state==[0,3]:
        state=[2,3]
        return state,5
    if action=='u':
        if state[0]==0:
            return state,-1
        else:
            state=[state[0]-1,state[1]]
            return state,0
    if action=='d':
        if state[0]==(GWsize-1):
            return state,-1
        else:
            state=[state[0]+1,state[1]]
            return state,0
    if action=='l':
        if state[1]==0:
            return state,-1
        else:
            state=[state[0],state[1]-1]
            return state,0
    if action=='r':
        if state[1]==(GWsize-1):
            return state,-1
        else:
            state=[state[0],state[1]+1]
            return state,0


GWsize=5
NumActions=4
V = np.zeros([GWsize, GWsize])  # V-values
P = np.array([0.25,0.25,0.25,0.25])             # Policy random at any state


Returns=[]
for i in range(GWsize):
    Returns.append([])
    for j in range(GWsize):
        Returns[i].append([])


state=[0,0]
Actions=['u','d','l','r']
gamma=0.9
theta=1e-4
EpisodeLength=10

def PrintV(v):
    for i in range(GWsize):
        for j in range(GWsize):
            print(f'{V[i][j]:.1f}',end=' ')
        print()
    print()

numberOfEpisodes=0
error=100000
while True:
    if numberOfEpisodes%1000==0:
        print("Number Of Episodes: ", numberOfEpisodes)
        print("Error: ", error)
    numberOfEpisodes+=1
    EpisodeRewards=[]
    CummAndDiscEpisodeRewards=[]    #  Gt
    state=[np.random.randint(GWsize),np.random.randint(GWsize)]
    ValuesOld=copy.deepcopy(V)
    for e in range(EpisodeLength):
        a=np.argmax(np.random.multinomial(1,P))
        newstate,rew = Environment(Actions[a],state)
        EpisodeRewards.append([state,rew,a])
        state=newstate

    for i in range(EpisodeLength):  # Compute Gt
        cum=0
        for j in range(i,EpisodeLength):
            cum+=np.power(gamma,j-i)*EpisodeRewards[j][1]
        CummAndDiscEpisodeRewards.append(cum)

    for i in range(EpisodeLength):  # Update
        EpisodeRewards[i][1]=CummAndDiscEpisodeRewards[i]


    CheckedStates=[]
    for i in range(EpisodeLength):  #  Append G to Returns
        state=EpisodeRewards[i][0]
        if not state in CheckedStates:   # First occurrence
            CheckedStates.append(state) 
            Returns[state[0]][state[1]].append(EpisodeRewards[i][1])
        V[state[0],state[1]]=np.average(np.array(Returns[state[0]][state[1]]))

    error=np.sum(np.abs(V-ValuesOld))
    if error<theta:
        PrintV(V)
        exit(0)


    
