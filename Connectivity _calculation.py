# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Importing necessary libraries
#------------------------------------------------------------------------------

import scipy.io as sio 
import os 
from mne.connectivity import spectral_connectivity
import numpy as np 
from nilearn.plotting import plot_matrix


#-----For Statistical Analysis
import scipy.stats as stats
#import researchpy as rp
#import statsmodels.api as sm
#from statsmodels.formula.api import ols
#import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
# Load the Data, Labels and define variables for Odor trials analysis
#------------------------------------------------------------------------------
srate = 1000
ntime   = 30*srate

path = r'C:\Users\lanan\Documents\Sleep Project\Sleep_Time_Frequency_Analysis\PythonConnectivity\ChannelOdor'

labels = sio.loadmat(os.path.join(path, 'labels.mat'))
labels = labels['labels'].tolist()[0]
labels = [label[0] for label in labels]
nscouts = len(labels)


subj = 0
connMatrixTotalOdorOn = np.zeros((nscouts,nscouts,21));
connMatrixTotalOdorOff = np.zeros((nscouts,nscouts,21));
connMatrixTotalOdorDiff = np.zeros((nscouts,nscouts,21));

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
        
        connMatrixOn = spectral_connectivity(data, method='coh', sfreq=srate, mode='multitaper', 
                                           fmin=1, fmax=4, faverage=1, tmin=15, tmax=29)
        connMatrixOn = np.squeeze(connMatrixOn[0])
        
        connMatrixOff = spectral_connectivity(data, method='coh', sfreq=srate, mode='multitaper', 
                                           fmin=1, fmax=4, faverage=1, tmin=0, tmax=15)
        connMatrixOff = np.squeeze(connMatrixOff[0])


        for i in range(nscouts):
            for j in range(nscouts):
                connMatrixOn[i,j] = connMatrixOn[j,i]
                connMatrixOff[i,j] = connMatrixOff[j,i]
                
        connMatrixTotalOdorOn[:,:,subj] = connMatrixOn
        connMatrixTotalOdorOff[:,:,subj] = connMatrixOff
        ConnMatrixDiffOdor = connMatrixOn - connMatrixOff
        
        #---- Plot Connectivity Matrices for each subject------
        #plot_matrix(connMatrixOn,title = (filename+'Odor On'),labels=labels, colorbar=True, tri='full', reorder=False, vmax=.8, vmin=0)
        #plot_matrix(connMatrixOff,title = (filename+'Odor Off'),labels=labels, colorbar=True, tri='full', reorder=False, vmax=.8, vmin=0)
        #plot_matrix(ConnMatrixDiffOdor,title = (filename+'Odor'),labels=labels,colorbar=True, tri='full', reorder=False, vmax=.05, vmin=0)
        
        connMatrixTotalOdorDiff[:,:,subj] = ConnMatrixDiffOdor
        subj += 1

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
connMatrixTotalPlaceboOn = np.zeros((nscouts,nscouts,21));
connMatrixTotalPlaceboOff = np.zeros((nscouts,nscouts,21));
connMatrixTotalPlaceboDiff = np.zeros((nscouts,nscouts,21));

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
        
        connMatrixOn = spectral_connectivity(data, method='coh', sfreq=srate, mode='multitaper', 
                                           fmin=1, fmax=4, faverage=1, tmin=15, tmax=29)
        connMatrixOn = np.squeeze(connMatrixOn[0])
        
        connMatrixOff = spectral_connectivity(data, method='coh', sfreq=srate, mode='multitaper', 
                                           fmin=1, fmax=4, faverage=1, tmin=0, tmax=15)
        connMatrixOff = np.squeeze(connMatrixOff[0])


        for i in range(nscouts):
            for j in range(nscouts):
                connMatrixOn[i,j] = connMatrixOn[j,i]
                connMatrixOff[i,j] = connMatrixOff[j,i]
                
        connMatrixTotalPlaceboOn[:,:,subj] = connMatrixOn
        connMatrixTotalPlaceboOff[:,:,subj] = connMatrixOff
        ConnMatrixDiffPlacebo = connMatrixOn - connMatrixOff
        
        #---- Plot Connectivity Matrices for each subject------
        #plot_matrix(connMatrixOn,title = (filename +'Placebo On'),labels=labels, colorbar=True, tri='full', reorder=False, vmax=.8, vmin=0)
        #plot_matrix(connMatrixOff,title = (filename +'Placebo Off'),labels=labels,  colorbar=True, tri='full', reorder=False, vmax=.8, vmin=0)
        #plot_matrix(ConnMatrixDiffOdor,title = (filename +'Placebo'),labels=labels, colorbar=True, tri='full', reorder=False, vmax=.05, vmin=0)
        
        connMatrixTotalPlaceboDiff[:,:,subj] = ConnMatrixDiffPlacebo
        subj += 1
        
#------------------------------------------------------------------------------
# Mean across subjects and Figures creation
#------------------------------------------------------------------------------
connMatrixMeanOdorOn = connMatrixTotalOdorOn.mean(2)
connMatrixMeanOdorOff = connMatrixTotalOdorOff.mean(2)
connMatrixMeanOdorDiff = connMatrixTotalOdorDiff.mean(2)

connMatrixMeanPlaceboOn = connMatrixTotalPlaceboOn.mean(2)
connMatrixMeanPlaceboOff = connMatrixTotalPlaceboOff.mean(2)
connMatrixMeanPlaceboDiff = connMatrixTotalPlaceboDiff.mean(2)


plot_matrix(connMatrixMeanOdorOn,title = 'mean subjects Odor On',labels=labels, colorbar=True, tri='full', reorder=False, vmax=.8, vmin=0)
plot_matrix(connMatrixMeanOdorOff,title = 'mean subjects Odor Off',labels=labels,  colorbar=True, tri='full', reorder=False, vmax=.8, vmin=0)
plot_matrix(connMatrixMeanOdorDiff,title = 'mean subjects Odor Diff',labels=labels,  colorbar=True, tri='full', reorder=False)#, vmax=.05, vmin=0)

plot_matrix(connMatrixMeanPlaceboOn,title = 'mean subjectsPlacebo On',labels=labels,  colorbar=True, tri='full', reorder=False, vmax=.8, vmin=0)
plot_matrix(connMatrixMeanPlaceboOff,title = 'mean subjects Placebo Off',labels=labels,  colorbar=True, tri='full', reorder=False, vmax=.8, vmin=0)
plot_matrix(connMatrixMeanPlaceboDiff,title = 'mean subjects Placebo Diff',labels=labels,  colorbar=True, tri='full', reorder=False)#, vmax=.05, vmin=0)


#------------------------------------------------------------------------------
#                       Statistical analysis
#------------------------------------------------------------------------------

statistics =  np.zeros((nscouts,nscouts))
pvalues = np.ones((nscouts,nscouts))
statisticsOdorVsOff =  np.zeros((nscouts,nscouts))
pvaluesOdorVsOff = np.ones((nscouts,nscouts))
statisticsDiff =  np.zeros((nscouts,nscouts))
pvaluesDiff = np.ones((nscouts,nscouts))

for chan1 in range(nscouts):
    for chan2 in range((chan1+1),nscouts):
        [statisticsDiff[chan1,chan2], pvaluesDiff[chan1,chan2]]= stats.ttest_ind(connMatrixTotalOdorDiff[chan1,chan2,:], connMatrixTotalPlaceboDiff[chan1,chan2,:])
        statisticsDiff[chan2,chan1]  = statisticsDiff[chan1,chan2]
        pvaluesDiff[chan2,chan1]  = pvaluesDiff[chan1,chan2]
        
        
        [statisticsOdorVsOff[chan1,chan2], pvaluesOdorVsOff[chan1,chan2]]= stats.ttest_ind(connMatrixTotalOdorOn[chan1,chan2,:],connMatrixTotalPlaceboOn[chan1,chan2,:])
        statisticsOdorVsOff[chan2,chan1]  = statisticsOdorVsOff[chan1,chan2]
        pvaluesOdorVsOff[chan2,chan1]  = pvaluesOdorVsOff[chan1,chan2]
        
        

plot_matrix(pvaluesDiff,labels=labels, colorbar=True, tri='full', reorder=False)
plot_matrix(pvaluesOdorVsOff,labels=labels, colorbar=True, tri='full', reorder=False)

   
significantconnectDiff = 1*(pvaluesDiff<= 0.05)
plot_matrix(significantconnectDiff,labels=labels, colorbar=True, tri='full', reorder=False)


significantconnectDiff = np.multiply(significantconnectDiff,pvaluesDiff)      
plot_matrix(significantconnectDiff,labels=labels, colorbar=True, tri='full', reorder=False)

significantconnectOdorVsOff = 1*(pvaluesOdorVsOff<= 0.05)
plot_matrix(significantconnectOdorVsOff,title='Significant Connect',labels=labels, colorbar=True, tri='full', reorder=False)
