#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:39:59 2019

@author: safeer
"""

import pandas as pd
import numpy as np
import scipy.sparse as scp
import random
import sys
import os

from math import sqrt
from sklearn.metrics import mean_squared_error

'''rating_BioData1 = scp.load_npz("/home/safeer/Documents/QCRI/QCRI-Deep-Learning/Codes-myVersion/neural-matrix-completion-master/Experiment2/BioDataset1/rating.npz").todense()
R_norm_BioData1 = np.subtract(scp.load_npz("/home/safeer/Documents/QCRI/QCRI-Deep-Learning/Codes-myVersion/neural-matrix-completion-master/Experiment2/BioDataset1/rating.npz").todense(),2)
recons_BioData1 = np.load("/home/safeer/Documents/QCRI/QCRI-Deep-Learning/Codes-myVersion/neural-matrix-completion-master/Experiment2/BioDataset1_output/recons_complete_matrix.npy")
mask_BioData1 = scp.load_npz("/home/safeer/Documents/QCRI/QCRI-Deep-Learning/Codes-myVersion/neural-matrix-completion-master/Experiment2/BioDataset1/train_mask.npz").todense()

R_norm = R_norm_BioData1
recons =recons_BioData1
mask = mask_BioData1'''

def output_csv(R_norm, recons, mask, nonmiss_perc, experiment_dir) :
    
    #selecting one random column for every 100 iterations 
    column_indices = []
    for i in range(0,1200,100) :
        
        column_index = int(i + random.random()*99)
        column_indices.append(column_index)
    
    
    RMSE_list = []
    correl_list = []
    
    R_clmns = []
    recons_clmns = []
    mask_clmns = []
    
    #extracting all columns from each matrix, calculating RMSE and correlation for rach pair of R and recons lists
    for i in range(12) :
        
        r_clmn = np.squeeze(np.asarray(R_norm[:, column_indices[i]]))
        recons_clmn = np.squeeze(np.asarray(recons[:, column_indices[i]]))
        mask_clmn = mask[:, column_indices[i]]
        
        R_clmns.append(r_clmn)
        recons_clmns.append(recons_clmn)
        mask_clmns.append(mask_clmn)
        
        RMSE_list.append(sqrt(mean_squared_error(r_clmn, recons_clmn)))
        correl_list.append(np.corrcoef(r_clmn, recons_clmn))
    
    
    #make dataframe
    output_dict = {}
    for i in range(12) :
        
        output_dict[str("R")+str(i)] = R_clmns[i]
        output_dict[str("mask")+str(i)] = np.squeeze(np.asarray(mask_clmns[i]))
        output_dict[str("recons")+str(i)] = recons_clmns[i]
        output_dict[str("RMSE")+str(i)] = RMSE_list[i]
        output_dict[str("correlation")+str(i)] = correl_list[i][0,1]
        output_dict[str('Space') + str(i)] = " "
    
    
    output_df = pd.DataFrame(output_dict)
    
    i = 3
    while i < 71 :
        output_df.iloc[1:,[i,i+1]] = np.nan
        i+=6
    
        
    if not os.path.exists("../" + experiment_dir + "/output_data") :
        os.mkdir("../" + experiment_dir + "/output_data")


    output_df.to_csv("../" + experiment_dir + "/output_data/BioData"+str(nonmiss_perc) + ".csv")


file_num = [1,3,9,18,45,91]

for i in file_num :
    #rating_BioData1 = scp.load_npz("/home/safeer/Documents/QCRI/QCRI-Deep-Learning/Codes-myVersion/neural-matrix-completion-master/Experiment2/BioDataset1/rating.npz").todense()
    R_norm = np.subtract(scp.load_npz("../" + sys.argv[1] + "/BioDataset" + str(i) + "/rating.npz").todense(),2)
    recons = np.load("../" + sys.argv[1] + "/BioDataset" + str(i)  + "_output/recons_complete_matrix.npy")
    mask = scp.load_npz("../" + sys.argv[1] + "/BioDataset" + str(i) + "/train_mask.npz").todense()
    
    output_csv(R_norm, recons, mask, i, sys.argv[1])
    
