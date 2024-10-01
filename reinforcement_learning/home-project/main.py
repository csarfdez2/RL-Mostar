#!/usr/bin/python
# -*- coding: utf-8 -*-
# Csar Fdez - 2024
# Reinforce for HomeSolar Environment
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
import argparse
import copy
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--train", action = 'store_true')
parser.add_argument("--trajectory", action = 'store_true')
parser.add_argument("--episode", default=0)
args = parser.parse_args()

def processData():
    dfarray=[]
    for i in range(1,7):
        csv_path="./data/power"+"{:02d}".format(i)+".csv"
        print("loading: ",csv_path)
        dfarray.append(pd.read_csv(csv_path))
        dfarray[i-1].fillna(value=0,inplace=True)
    return dfarray

def consumerProfile(l):
    # returns a DAILY consumer profile as a mean of a list of months (argument)
    # Data granularity is 30' -> 48 slots per day
    d = np.hstack(( [ dfarray[i-1]["C"].values/1000  for i in l ] ))
    lendata=d.shape[0]
    ret=np.zeros(48)
    for i in range(lendata//48):
        ret+=d[i*48:(i+1)*48]
    return ret/(lendata//48)

def handMadeConsumerProfile():
    ret=np.zeros(48)
    for i in range(48):
        ret[i]=0.2   # base consumption
    for i in range(16,20):
        ret[i]=3   # peak consumption
    for i in range(32,40):
        ret[i]=2.5  # peak consumption
    return ret

def solarPower(l):
    # returns solar power 
    d = np.hstack(( [ dfarray[i-1]["Generated"].values/1000  for i in l ] ))
    lendata=d.shape[0]
    ret=np.vstack(( d[i*48:(i+1)*48]  for i in range(lendata//48)    ))
    return ret

def chargeLevels():
    ret=[0]
    for i in range(N-1):
        ret.append((i+1)*step_power)
    for i in range(N-1):
        ret.append(-(i+1)*step_power)
    return ret

def Tariff():
    # creates buy/sell tariff
    sell=np.ones(48)*0.12   # euros/KwH
    valley=0.07
    flat=0.14
    peak=0.2
    buy=np.ones(48)*valley    # euros/KwH  valley,flat,peak
    for i in range(12*2,15*2):
        buy[i]=peak
    for i in range(18*2,20*2):
        buy[i]=peak
    for i in range(8*2,12*2):
        buy[i]=flat
    for i in range(20*2,24*2):
        buy[i]=flat
    return(buy,sell)

max_battery=10  # WH
step_power=1  # W
N=5
charge_levels = chargeLevels()

def Environment(state,action,episode):
    # state: slot, battery
    # item: from 0 to datalen
    # action: one shot 1+2*(N-1).  N levels of charge or discharge
    slot=state[0]
    battery=state[1]
    charge=(np.array(charge_levels)*np.array(action)).sum()
    max_to_battery=(max_battery-battery)
    max_from_battery=battery

    if charge==0:    # positive: battery storing energy. negative: providing
        battery_flow=0
    elif charge>0:
        battery_flow=+min(charge/2,max_to_battery) # 1/2 because it is 30'
    else:
        battery_flow=max(charge/2,-max_from_battery)
    nextbattery=battery+battery_flow
    from_grid=consumer_profile[slot]-solar_power[episode,slot]+battery_flow
    if from_grid>0:
        reward = -buy_tariff[slot]*from_grid
    else:
        reward = -sell_tariff[slot]*from_grid
    nextslot=(slot+1)%48
    #if nextslot==47:
    #    reward+=(nextbattery-init_battery)*sell_tariff[0]
    next_state=(nextslot, nextbattery)
    return (reward,next_state)

def oneHot(n,nmax):
    ret=np.zeros(nmax)
    ret[n]=1
    return ret

dfarray=processData()
months_to_define_consumer_profile=[1,2,3]
months_to_train=[1,2,3]
#consumer_profile=consumerProfile(months_to_define_consumer_profile)
consumer_profile=handMadeConsumerProfile()

solar_power=solarPower(months_to_train)
(buy_tariff,sell_tariff)=Tariff()
datalen=solar_power.shape[0]


input_dimension=48+1  # OneHot slots +  battery
num_actions=1+2*(N-1)
gamma=1
learning_rate_policy=1e-3

hidden_size=256

def ModelPolicy():
    m=nn.Sequential( nn.Linear(input_dimension, hidden_size), nn.ReLU(), nn.Linear(hidden_size, num_actions), nn.Softmax(dim=-1))
    return m

def predictPolicy(state):
    action_probs = modelPolicy(torch.FloatTensor(state))
    return action_probs

modelPolicy=ModelPolicy()
optimizerPolicy = optim.Adam(modelPolicy.parameters(), lr=learning_rate_policy)

def discount_rewards(rewards):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return r             

action_space = np.arange(num_actions)
batch_size=16

init_battery=0

def oneBatchREINFORCE(ep):
    global episodesLength,init_battery
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 0

    while batch_counter<batch_size:
        state=[0,init_battery]
        states = []
        rewards = []
        actions = []
        for step in range(48):
            action_probs = predictPolicy(np.hstack(( oneHot(state[0],48), state[1] ))  ).detach().numpy()
            try:  
                action = np.random.choice(action_space, p=action_probs)
            except:
                print("EXCEPT",ep,step)
                exit(0)
            reward, next_state= Environment(state,oneHot(action,num_actions),ep)
            states.append(  np.hstack(( oneHot(state[0],48), state[1] ))   )
            rewards.append(reward)
            actions.append(action)
            state=next_state
            #if step==47:
            #    init_battery=state[1]
        batch_rewards.extend(discount_rewards(rewards))
        batch_states.extend(states)
        batch_actions.extend(actions)
        batch_counter += 1
        episodesLength.append(sum(rewards))
        if batch_counter==(batch_size-1):
            optimizerPolicy.zero_grad()
            state_tensor = torch.FloatTensor(np.array(batch_states))
            reward_tensor =torch.FloatTensor(batch_rewards) 
            action_tensor = torch.LongTensor(batch_actions)
            # Compute loss
            logprob = torch.log(predictPolicy(state_tensor))
            selected_logprobs = reward_tensor * logprob[np.arange(len(action_tensor)), action_tensor]
            loss = -selected_logprobs.mean()
            loss.backward()
            optimizerPolicy.step()


NumEpTrain=2000
episodesLength=[]
convolveWindow=50
modelname1="Reinforcetorch"
dataname="ReinforcetorchepisodesLength.pk"


def Trajectory(ep):
    bat=[]
    rewardacum=[]
    modelPolicy.load_state_dict(torch.load(modelname1))
    state=[0,init_battery]
    rew=0
    for step in range(48):
        rewardacum.append(rew)
        action_probs = predictPolicy(np.hstack(( oneHot(state[0],48), state[1] ))  ).detach().numpy()
        action = np.random.choice(action_space, p=action_probs)
        reward, next_state= Environment(state,oneHot(action,num_actions),ep)
        rew+=reward 
        bat.append(state[1])
        state=next_state
        print(action)
    print("Reward: ",rew)
    return(bat,rewardacum)


def plotTrajectoryConsumption(ep,battery,reward):
    pl.rcParams.update({'font.size': 7})
    f,ax = pl.subplots(1,1,figsize=(6,7), dpi=200, facecolor='white', edgecolor='k',sharex=True)    
    ax.plot (consumer_profile,linewidth=2,label="Consumer Profile")
    ax.plot (solar_power[ep],linewidth=2,label="Solar produciton")
    ax.plot (battery,linewidth=2,label="Battery")

    ax.set_ylabel("Energy (KwH)")
    ax.set_xlabel("Slot")
    ax1=ax.twinx()
    ax1.plot(buy_tariff,linewidth=1,label="Buy Energy Tariff", color="red")
    ax1.plot(rew,linewidth=1,label="Reward", color="blue")
    ax1.set_ylabel("Euros")
    ax.grid()
    ax.legend(loc='upper left')
    ax1.legend(loc='upper right')
    pl.show()


if args.train:
    for ep in tqdm(range(NumEpTrain)):
        oneBatchREINFORCE(0)  # Train only for dataset 0
        #oneBatchREINFORCE(ep%datalen)

    filehandler = open(dataname, 'wb') 
    pickle.dump(episodesLength, filehandler)
    filehandler.close()
    torch.save(modelPolicy.state_dict(),modelname1)

elif args.trajectory:
    trajectoryEpisode=int(args.episode)
    bat,rew=Trajectory(trajectoryEpisode)
    plotTrajectoryConsumption(trajectoryEpisode,bat,rew)
else:
    filehandler = open(dataname, 'rb') 
    el = pickle.load(filehandler)
    elfiltered = np.convolve(np.array((el)), np.ones(convolveWindow)/convolveWindow, mode='valid')


    pl.rcParams.update({'font.size': 7})
    f,ax = pl.subplots(1,1,figsize=(6,7), dpi=200, facecolor='white', edgecolor='k',sharex=True)    
    ax.plot(    elfiltered,linewidth=2,label="REINFORCE HomeSolar ")

    ax.set_ylabel("Episode reward (averaged over last "+str(convolveWindow)+" episodes)")
    ax.set_xlabel("Number of training episodes")
    pl.grid()
    pl.legend(loc='lower right')
    pl.show()


