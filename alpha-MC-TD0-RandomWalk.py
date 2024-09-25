#!/usr/bin/python
# -*- coding: utf-8 -*-
# Csar Fdez - 2024
# alpha-MC and alpha-TD0   for Random-Walk environ 

import numpy as np
import copy
import pylab as pl

def Environment(state):
    # Return -1 when Finish
    action=np.random.randint(2)  # 0:left,  1:right
    next_state=state+(2*action-1)
    rew=0
    done=False
    if next_state>4:
        rew=1
        next_state=-1
    if next_state<0:
        next_state=-1

    return next_state,rew

def RMS(v):
    return np.sqrt(np.average(np.power(v-TrueValues,2)))

def alphaMC(alpha):
    Rarray=np.zeros((MaxEpisodes))
    for i in range(NumRepetitions):
        R=[]
        V=np.zeros([5])+0.5 # Init values at 0.5
        episode=0
        while episode<MaxEpisodes:    
            EpisodeRewards=[]
            CummAndDiscEpisodeRewards=[]    #  Gt
            StatesInEpisode=[]
            state=3
            StatesInEpisode.append(state)    
            while True:
                next_state,rew = Environment(state)
                EpisodeRewards.append([state,rew])        
                state=next_state
                if state==-1:   # End of episode
                    EpisodeLength=len(EpisodeRewards)
                    for i in range(EpisodeLength):  # Compute Gt
                        cum=0
                        for j in range(i,EpisodeLength):
                            cum+=EpisodeRewards[j][1]
                        CummAndDiscEpisodeRewards.append(cum)
                    # Update Values
                    CheckedStates=[]
                    for i in range(EpisodeLength):
                        state=EpisodeRewards[i][0]              
                        if not  state in CheckedStates:
                            CheckedStates.append(state) 
                            V[state]=V[state]+alpha*(CummAndDiscEpisodeRewards[i]-V[state])

                    break
                else:
                    if not state in StatesInEpisode:
                        StatesInEpisode.append(state)
            Error=RMS(V)
            episode+=1
            R.append(Error)
        Rarray+=np.array((R))
    return Rarray/NumRepetitions


def alphaTD0(alpha):
    Rarray=np.zeros((MaxEpisodes))
    for i in range(NumRepetitions):
        R=[]
        V=np.zeros([5])+0.5  # Init values at 0.5
        episode=0
        while episode<MaxEpisodes:    
            state=3
            while True:
                next_state,rew = Environment(state)
                if next_state==-1 and state==4 :   # End of episode by right
                    V[state]=V[state]+alpha*(rew-V[state])
                    break
                elif  next_state==-1 and state==0:  # End of episode by left
                    V[state]=V[state]+alpha*(rew-V[state])
                    break
                else:
                    V[state]=V[state]+alpha*(rew+V[next_state]-V[state])
                    state=next_state


            Error=RMS(V)
            episode+=1
            R.append(Error)
        Rarray+=np.array((R))
    return Rarray/NumRepetitions


TrueValues=np.array([1/6,2/6,3/6,4/6,5/6],dtype=float)
MaxEpisodes=200
NumRepetitions=100

ResultsMC=[]
ResultsTD=[]
AlphasMC=[0.01,0.03,0.07]
AlphasTD=[0.05,0.1,0.15]


for a in AlphasTD:
    ResultsTD.append(alphaTD0(a))
for a in AlphasMC:
    ResultsMC.append(alphaMC(a))

pl.rcParams.update({'font.size': 7})
f,ax = pl.subplots(1,1,figsize=(6,7), dpi=200, facecolor='white', edgecolor='k',sharex=True)    

for i in range(len(AlphasMC)):
    color='C'+str(i)
    ax.plot(ResultsMC[i],color=color,linewidth=1,label=str(AlphasMC[i])+"-MC")

for i in range(len(AlphasTD)):
    color='C'+str(i)
    ax.plot(ResultsTD[i],color=color,linestyle='--',linewidth=1,label=str(AlphasTD[i])+"-TD")

ax.set_ylabel("RMS")
ax.set_xlabel("Episode")
pl.grid()
pl.legend(loc='upper right')
pl.show()



