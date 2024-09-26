#!/usr/bin/python
# -*- coding: utf-8 -*-
# Csar Fdez - 2024
# Stocastic Gradient Prediction for Random Walk

import numpy as np
import copy
import pylab as pl
from tqdm import tqdm

NumStates=21

def Environment(state):
    action=np.random.randint(2)  # 0:left,  1:right
    next_state=state+(2*action-1)
    rew=0
    done=False
    if next_state>=(NumStates-1):
        rew=1
        next_state=-1
    if next_state<0:
        next_state=-1

    return next_state,rew

def RMS(v):
    return np.sqrt(np.average(np.power(v-TrueValues,2)))
    #return np.sum(np.power(np.abs(v-TrueValues),2))/NumStates

def alphaMC(alpha):
    Rarray=np.zeros((MaxEpisodes))
    for i in range(NumRepetitions):
        R=[]
        V=np.zeros([NumStates]) 
        episode=0
        while episode<MaxEpisodes:    
            EpisodeRewards=[]
            CummAndDiscEpisodeRewards=[]    #  Gt
            StatesInEpisode=[]
            state=int(NumStates/2)
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


def X(s):
    ret=np.zeros([NumStates])
    ret[s]=1
    return ret


def stochasticGradient(alpha):
    Rarray=np.zeros((MaxEpisodes))
    for i in range(NumRepetitions):    
        R=[]
        V=np.zeros([NumStates])  
        w=np.zeros([NumStates])
        episode=0
        while episode<MaxEpisodes:    
            EpisodeRewards=[]
            CummAndDiscEpisodeRewards=[]    #  Gt
            StatesInEpisode=[]
            state=int(NumStates/2)
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
                    for i in range(EpisodeLength):
                        w+=alpha*(CummAndDiscEpisodeRewards[i]-np.dot(w,X(EpisodeRewards[i][0])))*X(EpisodeRewards[i][0])
                    break
                else:
                    if not state in StatesInEpisode:
                        StatesInEpisode.append(state)

            # Update values for each state
            for state in range(NumStates):
                V[state]=np.dot(w,X(state))

            Error=RMS(V)
            episode+=1
            R.append(Error)
        Rarray+=np.array((R))            
    return Rarray/NumRepetitions


def stochasticSemiGradientTD0(alpha):
    Rarray=np.zeros((MaxEpisodes))
    for i in range(NumRepetitions):
        R=[]
        V=np.zeros([NumStates])    
        w=np.zeros([NumStates])

        episode=0
        while episode<MaxEpisodes:    
            state=int(NumStates/2)
            while True:
                next_state,rew = Environment(state)
                if next_state==-1 and state==(NumStates-1) :   # End of episode by right
                    w+=alpha*(rew -np.dot(w,X(state))  )*X(state)
                    break
                elif  next_state==-1 and state==0:  # End of episode by left
                    w+=alpha*(rew -np.dot(w,X(state))  )*X(state)
                    break
                else:
                    w+=alpha*(rew+np.dot(w,X(next_state)) -np.dot(w,X(state))  )*X(state)
                    state=next_state

            # Update values for each state
            for state in range(NumStates):
                V[state]=np.dot(w,X(state))

            Error=RMS(V)
            episode+=1
            R.append(Error)
        Rarray+=np.array((R))            
    return Rarray/NumRepetitions


TrueValues=np.zeros((NumStates),dtype=float)
for i in range(NumStates):
    TrueValues[i]=i/(NumStates-1)
MaxEpisodes=400
NumRepetitions=50

ResultsMC=[]
ResultsSG=[]
ResultsSGTD=[]
Alphas=[0.05,0.01,0.005]
AlphasSGTD=[0.3,0.2,0.1]

for a in tqdm(Alphas):
    ResultsMC.append(alphaMC(a))
    ResultsSG.append(stochasticGradient(a))

for a in tqdm(AlphasSGTD):
    ResultsSGTD.append(stochasticSemiGradientTD0(a))

pl.rcParams.update({'font.size': 7})
f,ax = pl.subplots(1,1,figsize=(6,7), dpi=200, facecolor='white', edgecolor='k',sharex=True)    

for i in range(len(Alphas)):
    color='C'+str(i)
    ax.plot(ResultsMC[i],color=color,linewidth=1,label=str(Alphas[i])+"-MC")
    ax.plot(ResultsSG[i],color=color,linewidth=1,linestyle='--',label=str(Alphas[i])+"-SG")
ax.set_ylabel("RMS")
ax.set_xlabel("Episode")
pl.grid()
pl.legend(loc='upper left')
pl.title('Random Walk '+str(NumStates)+' states')
pl.show()

pl.rcParams.update({'font.size': 7})
f,ax = pl.subplots(1,1,figsize=(6,7), dpi=200, facecolor='white', edgecolor='k',sharex=True)    

for i in range(len(Alphas)):
    color='C'+str(i)
    ax.plot(ResultsMC[i],color=color,linewidth=1,label=str(Alphas[i])+"-MC")
for i in range(len(AlphasSGTD)):
    color='C'+str(i)
    ax.plot(ResultsSGTD[i],color=color,linewidth=1,linestyle='--',label=str(AlphasSGTD[i])+"-SGTD0")
ax.set_ylabel("RMS")
ax.set_xlabel("Episode")
pl.grid()
pl.legend(loc='upper left')
pl.title('Random Walk '+str(NumStates)+' states')
pl.show()

