#!/usr/bin/python
# -*- coding: utf-8 -*-
# Csar Fdez - 2024
#  Value iteration for optimal policy
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

def printArrow(actions):
    for a in actions:
        if a=='l':
            print('\u2190 ',end='')
        elif a=='u':
            print('\u2191 ',end='')
        elif a=='r':
            print('\u2192 ',end='')
        elif a=='d':
            print('\u2193 ',end='')
        else:
            print("Error in action:", a)
            exit(0)
    print()


GWsize=5
Values = np.zeros([GWsize, GWsize])
BestAction=[]
for i in range(GWsize):
    BestAction.append([])
    for j in range(GWsize):
        BestAction[i].append('')

state=[0,0]
Actions=['u','d','l','r']
gamma=0.9
theta=1e-2

while True:
    delta=0
    for i in range(GWsize):
        for j in range(GWsize):
            val=0
            ValuesOld=copy.deepcopy(Values)
            StateActionValues=np.zeros(4)
            for a in range(4):
                state=[i,j]
                state,rew = Environment(Actions[a],state)
                StateActionValues[a]=(rew+gamma*Values[state[0],state[1]])
            action=np.argmax(StateActionValues)
            Values[i,j]=StateActionValues[action]
            BestAction[i][j]=Actions[action]
            delta=max(delta,abs(Values[i,j]-ValuesOld[i,j]))

    if delta<theta:
        with np.printoptions(precision=1, suppress=True):
            print(Values)
    
        print()
        for i in range(GWsize):
            printArrow(BestAction[i])
        exit(0)


    
