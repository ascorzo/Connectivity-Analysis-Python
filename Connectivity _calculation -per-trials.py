# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Importing necessary libraries
#------------------------------------------------------------------------------

import scipy.io as sio 
import os 
from mne.connectivity import spectral_connectivity
import numpy as np 
from nilearn.plotting import plot_matrix
import math

#-----For Statistical Analysis
#import scipy.stats as stats
#import researchpy as rp
#import statsmodels.api as sm
#from statsmodels.formula.api import ols
#import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
# Load the Data, Labels and define variables for Odor trials analysis
#------------------------------------------------------------------------------
srate = 1000
ntime   = 30*srate

freqMin = 1
freqMax = 4

path = r'C:\Users\lanan\Documents\Sleep Project\Sleep_Time_Frequency_Analysis\PythonConnectivity\ChannelOdor'

labels = sio.loadmat(os.path.join(path, 'labels.mat'))
labels = labels['labels'].tolist()[0]
labels = [label[0] for label in labels]
nscouts = len(labels)


subj = 0
mintrials = 100
connMatrixTotalOdorOn_trials = np.zeros((nscouts,nscouts,mintrials,21));
connMatrixTotalOdorOff_trials = np.zeros((nscouts,nscouts,mintrials,21));
connMatrixTotalOdorDiff_trials = np.zeros((nscouts,nscouts,mintrials,21));



for filename in os.listdir(path):
    if filename.endswith(".mat") and filename.startswith("S"):
        print(filename)
    
        data = sio.loadmat(os.path.join(path,filename))
        data = data['data']

        #----------------------------------------------------------------------
        # Connectivity Analysis
        #----------------------------------------------------------------------
        data = data.reshape([nscouts, -1, ntime])
        #plt.plot(data[1][:][3])
        data = data.transpose([1,0,2])
        
        trials = data.shape[0] 
        mintrials = min(trials,mintrials)
        
        connMatrixOn_trials = np.zeros((nscouts,nscouts,math.floor(trials/5)));
        connMatrixOff_trials = np.zeros((nscouts,nscouts,math.floor(trials/5)));
        connMatrixDiff_trials = np.zeros((nscouts,nscouts,math.floor(trials/5)));
        
        for trial in range(math.floor(trials/5)):
            dataTemp = data[trial*5:(trial+1)*5-1,:,:]
            connMatrixOn = spectral_connectivity(dataTemp, method='coh', sfreq=srate, mode='multitaper', 
                                           fmin=freqMin, fmax=freqMax, faverage=1, tmin=15, tmax=29)
            connMatrixOn = np.squeeze(connMatrixOn[0])
            
            
            connMatrixOff = spectral_connectivity(dataTemp, method='coh', sfreq=srate, mode='multitaper', 
                                           fmin=freqMin, fmax=freqMax, faverage=1, tmin=0, tmax=15)
            connMatrixOff = np.squeeze(connMatrixOff[0])
            
            for i in range(nscouts):
                for j in range(nscouts):
                    connMatrixOn[i,j] = connMatrixOn[j,i]
                    connMatrixOff[i,j] = connMatrixOff[j,i]
                    
                    
            connMatrixOn_trials[:,:,trial] = connMatrixOn
            connMatrixOff_trials[:,:,trial] = connMatrixOff
            connMatrixDiff_trials[:,:,trial] = connMatrixOn-connMatrixOff
            
            #---- Plot Connectivity Matrices for each subject------
#            plot_matrix(connMatrixOn,title = (filename+'Odor On'),labels=labels, colorbar=True, tri='full', reorder=False)#, vmax=.8, vmin=0)
#            plot_matrix(connMatrixOff,title = (filename+'Odor Off'),labels=labels, colorbar=True, tri='full', reorder=False)#, vmax=.8, vmin=0)
#            plot_matrix(connMatrixOn-connMatrixOff,title = (filename+'Odor'),labels=labels,colorbar=True, tri='full', reorder=False)#, vmax=.05, vmin=0)
#            
        connMatrixTotalOdorOn_trials[:,:,:,subj] = connMatrixOn_trials
        connMatrixTotalOdorOff_trials[:,:,:,subj] = connMatrixOff_trials
        connMatrixTotalOdorDiff_trials[:,:,:,subj] = connMatrixDiff_trials

        subj += 1

connMatrixTotalOdorOn_trials= connMatrixOn_trials[:,:,1:mintrials,:]
connMatrixTotalOdorOff_trials = connMatrixOff_trials[:,:,1:mintrials,:]
connMatrixTotalOdorDiff_trials = connMatrixDiff_trials[:,:,1:mintrials,:]
#------------------------------------------------------------------------------
# Load the Data, Labels and define variables for Placebo trials analysis
#------------------------------------------------------------------------------


srate = 1000
ntime   = 30*srate

path = r'C:\Users\lanan\Documents\Sleep Project\Sleep_Time_Frequency_Analysis\PythonConnectivity\ChannelPlacebo'

labels = sio.loadmat(os.path.join(path, 'labels.mat'))
labels = labels['labels'].tolist()[0]
labels = [label[0] for label in labels]
nscouts = len(labels)


subj = 0
mintrials = 100
connMatrixTotalPlaceboOn_trials = np.zeros((nscouts,nscouts,mintrials,21));
connMatrixTotalPlaceboOff_trials = np.zeros((nscouts,nscouts,mintrials,21));
connMatrixTotalPlaceboDiff_trials = np.zeros((nscouts,nscouts,mintrials,21));


for filename in os.listdir(path):
    if filename.endswith(".mat") and filename.startswith("S"):
        print(filename)
    
        data = sio.loadmat(os.path.join(path,filename))
        data = data['data']

        #----------------------------------------------------------------------
        # Connectivity Analysis
        #----------------------------------------------------------------------
        data = data.reshape([nscouts, -1, ntime])
        #plt.plot(data[1][:][3])
        data = data.transpose([1,0,2])
        
        trials = data.shape[0] 
        
        connMatrixOn_trials = np.zeros((nscouts,nscouts,math.floor(trials/5)));
        connMatrixOff_trials = np.zeros((nscouts,nscouts,math.floor(trials/5)));
        connMatrixDiff_trials = np.zeros((nscouts,nscouts,math.floor(trials/5)));
        
        for trial in range(math.floor(trials/5)):
            dataTemp = data[trial*5:(trial+1)*5-1,:,:]
            connMatrixOn = spectral_connectivity(dataTemp, method='coh', sfreq=srate, mode='multitaper', 
                                           fmin=freqMin, fmax=freqMax, faverage=1, tmin=15, tmax=29)
            connMatrixOn = np.squeeze(connMatrixOn[0])
            
            
            connMatrixOff = spectral_connectivity(dataTemp, method='coh', sfreq=srate, mode='multitaper', 
                                           fmin=freqMin, fmax=freqMax, faverage=1, tmin=0, tmax=15)
            connMatrixOff = np.squeeze(connMatrixOff[0])
            
            for i in range(nscouts):
                for j in range(nscouts):
                    connMatrixOn[i,j] = connMatrixOn[j,i]
                    connMatrixOff[i,j] = connMatrixOff[j,i]
                    
                    
            connMatrixOn_trials[:,:,trial] = connMatrixOn
            connMatrixOff_trials[:,:,trial] = connMatrixOff
            connMatrixDiff_trials[:,:,trial] = connMatrixOn-connMatrixOff
            
            #---- Plot Connectivity Matrices for each subject------
#            plot_matrix(connMatrixOn,title = (filename+'Placebo On'),labels=labels, colorbar=True, tri='full', reorder=False)#, vmax=.8, vmin=0)
#            plot_matrix(connMatrixOff,title = (filename+'Placebo Off'),labels=labels, colorbar=True, tri='full', reorder=False)#, vmax=.8, vmin=0)
#            plot_matrix(connMatrixOn-connMatrixOff,title = (filename+'Placebo'),labels=labels,colorbar=True, tri='full', reorder=False)#, vmax=.05, vmin=0)
#            
            connMatrixTotalPlaceboOn_trials[:,:,:,subj] = connMatrixOn_trials
            connMatrixTotalPlaceboOff_trials[:,:,:,subj] = connMatrixOff_trials
            connMatrixTotalPlaceboDiff_trials[:,:,:,subj] = connMatrixDiff_trials
            
        subj += 1
        
connMatrixTotalOdorOn_trials= connMatrixOn_trials[:,:,1:mintrials,:]
connMatrixTotalOdorOff_trials = connMatrixOff_trials[:,:,1:mintrials,:]
connMatrixTotalOdorDiff_trials = connMatrixDiff_trials[:,:,1:mintrials,:]