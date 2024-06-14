#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" running the actual simulation 
    Deyue Kong and Marina Wosniack, using Monica's and Varsha's previous codes.
    - refactored and extended by Jan Kirchner in May 2021
"""
#%% importing packages
import numpy as np
import random
#%%
def print_progress(t, T):
    print("{:.2%} finished.".format(t/T))
#%%
def runSimulation(prms , mats):

    # Initializing the activities
    v1 , s1 , rl = np.zeros([prms['N_v1'], 1]) , np.zeros([prms['N_s1'], 1]) , np.zeros([prms['N_rl'], 1])
    
    # generating shared and independent events
    total_ms = prms['total_ms']
    
    p1 = np.random.rand(total_ms , 1) <= (1./prms['L_p_v1'])*(1 - prms['spatio_temp_corr'])
    p2 = np.random.rand(total_ms , 1) <= (1./prms['L_p_s1'])*(1 - prms['spatio_temp_corr'])
    p3 = np.random.rand(total_ms , 1) <= (2/(prms['L_p_v1']+prms['L_p_v1']))*(prms['spatio_temp_corr'])
    pall = p1 + p2 + p3
    
    # initialize duration counters and accumulators
    L_dur_v1_counter = round(np.random.normal(prms['L_dur_v1'], prms['L_dur_v1'] * 0.1))
    L_dur_s1_counter = round(np.random.normal(prms['L_dur_s1'], prms['L_dur_s1'] * 0.1))
    W_vec_v1 , W_vec_s1 , vec_v1 , vec_s1 , vec_rl = [] , [] , [] , []  , []
    
    # initialize time vector
    time_vec_list = np.arange(0, total_ms, round(total_ms/prms['store_points']))#25.0
    
    # main loop over time     
    for tt in range(0, int(total_ms)):
        if ((tt+1) % int(total_ms/10) == 0) and prms['logFlag']:  print_progress(tt,int(total_ms))
        # generate local event
        if pall[tt] >= 1:
            # correlated events
            corrFlag = p3[tt] == 1; v1flag = p1[tt] == 1;
                
            # L-events in v1 and s1
            L_start_v1 = np.random.choice(prms['N_v1'], 1)[0]
            L_start_s1 = np.random.choice(prms['N_s1'], 1)[0]
            if corrFlag: L_start_s1 = L_start_v1
            
            L_len_v1 = random.randrange(int(prms['N_v1']*prms['L_range_v1'][0]), int(prms['N_v1']*prms['L_range_v1'][1] + 1))
            L_len_s1 = random.randrange(int(prms['N_s1']*prms['L_range_s1'][0]), int(prms['N_s1']*prms['L_range_s1'][1] + 1))
            
            eventID_v1 = np.mod(range(L_start_v1, L_start_v1 + L_len_v1), prms['N_v1']);
            eventID_s1 = np.mod(range(L_start_s1, L_start_s1 + L_len_s1), prms['N_s1']);
            
            if corrFlag: 
                eventID_s1 = eventID_v1;
                v1 = np.zeros([prms['N_v1'], 1]); s1 = np.zeros([prms['N_s1'], 1]);
                v1[eventID_v1, 0] = prms['L_amp_v1']; 
                s1[eventID_s1, 0] = prms['L_amp_s1'];
                L_dur_v1_counter = round(np.random.normal(prms['L_dur_v1'], prms['L_dur_v1'] * 0.1))
                L_dur_s1_counter = L_dur_v1_counter;
            elif v1flag: 
                v1 = np.zeros([prms['N_v1'], 1]);
                v1[eventID_v1, 0] = prms['L_amp_v1'];
                L_dur_v1_counter = round(np.random.normal(prms['L_dur_v1'], prms['L_dur_v1'] * 0.1))
            else: 
                s1 = np.zeros([prms['N_s1'], 1]);
                s1[eventID_s1, 0] = prms['L_amp_s1'];
                L_dur_s1_counter = round(np.random.normal(prms['L_dur_s1'], prms['L_dur_s1'] * 0.1))
                
        # updating the output
        rl = rl + (prms['dt'] / prms['tau_out']) * (-rl + np.dot(mats['W_v1'], v1) + np.dot(mats['W_s1'], s1))
        # updating the weights...
        if prms["Hebbflag"] == 1:
            mats['W_v1'] = mats['W_v1'] + (prms['dt'] / prms['tau_w']) * np.dot(rl, np.transpose(v1 - prms['corr_thres']))
            mats['W_s1'] = mats['W_s1'] + (prms['dt'] / prms['tau_w']) * np.dot(rl, np.transpose(s1 - prms['corr_thres']))

        if prms['bounded'] == True:
            mats['W_v1'] = np.maximum(np.minimum(mats['W_v1'] , prms['W_thres'][1]),prms['W_thres'][0])
            mats['W_s1'] = np.maximum(np.minimum(mats['W_s1'] , prms['W_thres'][1]),prms['W_thres'][0])
            
        # this is where the old bug was
        L_dur_v1_counter -= 1; L_dur_s1_counter -= 1;
        if L_dur_v1_counter == 0: v1 = np.zeros([prms['N_v1'], 1])
        if L_dur_s1_counter == 0: s1 = np.zeros([prms['N_s1'], 1])

        # I save the matrices only every 10000 time steps
        if tt in time_vec_list: 
            W_vec_v1.append(mats['W_v1']); W_vec_s1.append(mats['W_s1']); 
            vec_v1.append(v1); vec_s1.append(s1); vec_rl.append(rl);
        
    W_vec_v1 = np.array(W_vec_v1); W_vec_s1 = np.array(W_vec_s1);

    return time_vec_list, W_vec_v1, W_vec_s1 , prms , mats , vec_v1 , vec_s1 , vec_rl
