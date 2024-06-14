#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" The goal here is to sketch the code for the co-refinement of V1 and S1 via their interaction through RL
    Big questions here: emergence of multisensory cells... like one RL cell being sensitive to both a neighborhood of v1 and s1 neurons...
    If S1-RL is already refined, how this would influence the refinement between V1 and RL? 
    Deyue Kong and Marina Wosniack, using Monica's and Varsha's previous codes.
    - refactored and extended by Jan Kirchner in May 2021
    - refactored by JaeAnn Dwulet November 2023
"""
#%%
import sys
sys.path.append('/Users/dwuletj/Documents/GitHub_Gjorgjieva/Multimodal/')
print(sys.path)

import pickle ,json, time , datetime , numpy , multiprocessing , random, pathlib, re
from importlib import reload
from tools.biased_weights import biased_weights
from tools.runSimulation import runSimulation
# =============================================================================
import analysis.making_figures
reload(analysis.making_figures)
from analysis.making_figures import making_figures
import analysis.making_stats
reload(analysis.making_stats)
from analysis.making_stats import making_stats
# =============================================================================
#%%
def getPrmsMats():
    # load all the other parameters from a text file
    with open('parameters.txt') as f: prms = json.load(f)
    # generate biased weight matrices
    mats = dict()
    mats['W_v1'] = biased_weights(prms['N_v1'], prms['W_initial_v1'], prms['bias_v1'], prms['spread_v1'])
    mats['W_s1'] = biased_weights(prms['N_s1'], prms['W_initial_s1'], prms['bias_s1'], prms['spread_s1'])
    return prms , mats

# %%
simname = 'Test_'
numpy.random.seed(random.randint(1,99999999))
prms , mats = getPrmsMats()
# set the output path
prms['path'] = pathlib.Path('simulations/monte_carlo')
idfilestring = str(datetime.datetime.now()) + '_' + str(round(numpy.random.rand() , 5))
idfilereformat = re.sub("[ :,.]", "", idfilestring)

prms['path_main'] = prms['path'] / ('sim_' + simname + idfilereformat)
pathlib.Path(prms['path_main']).mkdir(parents=True, exist_ok=True)
# Monte Carlo sample parameters
#prms['corr_thres'] = .55 #numpy.random.uniform(low=0.25,high=0.75) # 0.5 # 
#prms['spatio_temp_corr'] = .3 #numpy.random.uniform(low=0.0,high=1.0)
prms['logFlag'] =  0
# run the simulation
simOutput = runSimulation(prms , mats)
#print(type(simOutput))
savefilename = prms['path_main'] / ('simOutput_' + simname + '.pickle')
with open(savefilename, 'wb') as f:
    pickle.dump(simOutput, f)

making_figures(simOutput[3]['path_main'], simOutput[3]['W_thres'],
        simOutput[4]['W_v1'], simOutput[4]['W_s1'],
        simOutput[1], simOutput[2], simOutput[0], prms['store_points'])
making_stats(simOutput[3]['path_main'], simOutput[4]['W_v1'], simOutput[4]['W_s1'],
    simOutput[3]['W_thres'][1]/5, simOutput[3]['N_v1'])

# %%
