#!/usr/bin/python
# -*- coding: utf-8 -*-
# Csar Fdez - 2024
## Monte Carlo On-policy First visit MC-control

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
Q = np.random.uniform(size=[GWsize, GWsize,NumActions])  # Q-values
P = []             # Policy
for i in range(GWsize):
    P.append([])
    for j in range(GWsize):
        P[i].append([0.25,0.25,0.25,0.25])

Returns=[]
for i in range(GWsize):
    Returns.append([])
    for j in range(GWsize):
        Returns[i].append([])
        for k in range(NumActions):
            Returns[i][j].append([])


state=[0,0]
Actions=['u','d','l','r']
gamma=0.9
delta=1e-5
EpisodeLength=100
eps=1e-1
NumberOfIdenticalIterationsToExit=1000


def getArrow(a):
    if Actions[a]=='l':
        return('\u2190')
    elif Actions[a]=='u':
        return('\u2191')
    elif Actions[a]=='r':
        return('\u2192')
    elif Actions[a]=='d':
        return('\u2193')
    else:
        print("Error in action:", a)
        exit(0)


def printPolicy():
    for i in range(GWsize):
        for j in range(GWsize):
            a=np.argmax(P[i][j])
            print(getArrow(a)," ",end='')
        print()

def UpdatePolicy(s,a):
    global P
    for i in range(NumActions):
        if i==a:
            P[s[0]][s[1]][i]=1-eps+eps/NumActions
        else:
            P[s[0]][s[1]][i]=eps/NumActions

    
def ComputeAndPrintStateValues():        
    Values = np.zeros([GWsize, GWsize])
    theta=1e-5
    while True:
        for i in range(GWsize):
            for j in range(GWsize):
                val=0
                ValuesOld=copy.deepcopy(Values)
                state=[i,j]
                #print(state,end='  ')
                if np.random.uniform() < 0.01:
                    a=np.random.randint(NumActions)
                else:
                    a=np.argmax(P[state[0]][state[1]])
                state,rew = Environment(Actions[a],state)
                #print(Actions[a],state,rew)
                Values[i,j]=rew+gamma*Values[state[0],state[1]]
        if np.sum(np.abs(Values-ValuesOld))<theta:
            with np.printoptions(precision=1, suppress=True):
                print(Values)
            return

count=0
while True:
    POld = copy.deepcopy(P)
    count+=1
    EpisodeRewards=[]
    CummAndDiscEpisodeRewards=[]    #  Gt
    StatesInEpisode=[]
    state=[np.random.randint(GWsize),np.random.randint(GWsize)]
    StatesInEpisode.append(state)
    for e in range(EpisodeLength):
        a=np.argmax(np.random.multinomial(1,P[state[0]][state[1]]))
        newstate,rew = Environment(Actions[a],state)
        EpisodeRewards.append([state,a,rew])
        state=newstate
        if not newstate in StatesInEpisode:
            StatesInEpisode.append(newstate)

    for i in range(EpisodeLength):  # Compute Gt
        cum=0
        for j in range(i,EpisodeLength):
            cum+=np.power(gamma,j-i)*EpisodeRewards[j][2]
        CummAndDiscEpisodeRewards.append(cum)

    for i in range(EpisodeLength):  # Update
        EpisodeRewards[i][2]=CummAndDiscEpisodeRewards[i]

    CheckedStates=[]
    for i in range(EpisodeLength):  #  Append G to Returns
        state=EpisodeRewards[i][0]
        action=EpisodeRewards[i][1]
        if not  state in CheckedStates:
            CheckedStates.append(state) 
            Returns[state[0]][state[1]][action].append(EpisodeRewards[i][2])
        Q[state[0],state[1],action]=np.average(np.array(Returns[state[0]][state[1]][action]))

    # Update policy
    for s in StatesInEpisode:
        Astar=np.argmax(Q[s[0],s[1],:])
        UpdatePolicy(s,Astar)

    if np.array_equal(POld,P):
        if count>NumberOfIdenticalIterationsToExit:
            ComputeAndPrintStateValues()
            printPolicy()
            exit(0)
    else:
        count=0



    
