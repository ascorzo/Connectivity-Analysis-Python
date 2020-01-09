# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Importing necessary libraries
#------------------------------------------------------------------------------

import scipy.io as sio 
import os 
from mne.connectivity import spectral_connectivity
import numpy as np 
from nilearn.plotting import plot_matrix, view_connectome, show


#-----For Statistical Analysis
import scipy.stats as stats
#import researchpy as rp
#import statsmodels.api as sm
#from statsmodels.formula.api import ols
#import matplotlib.pyplot as plt



#------------------------------------------------------------------------------
# CONFIG
#------------------------------------------------------------------------------
channelOdorPath = r'data\ChannelOdor'
channelPlaceboPath = r'data\ChannelPlacebo'
calculationsSavingPath = r'data\calculated'
channelLocationsPath = r'data\channlocsMNI2.txt'
save_calculations = True
load_calculations = True
edge_threshold = "95%" #thresshold to draw an edge in connectome viz


#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
def read_channLoc(channelLocationsPath):
    coords = np.zeros((128,3))
    with open(channelLocationsPath, "r") as fp:
        line = fp.readline()
        index = 0
    while index<128:
       name, _, x, y, z = line.strip().split()
    #    x = float(x)
    #    y = float(y)
    #    z = float(z)
       coords[index] = np.array([x,y,z])
       line = fp.readline()
       index += 1
    return coords

def read_channLocMNI(channelLocationsPath):
    coords = np.zeros((128,3))
    with open(channelLocationsPath, "r") as fp:
        line = fp.readline()
        index = 0
        while index<128:
            name, x, y, z = line.strip().split()
            # x, z, y, name = line.strip().split()
            # z, x, y, name = line.strip().split()
            # y, x, z, name = line.strip().split()
            # z, y, x, name = line.strip().split()
            # y, z, x, name = line.strip().split()
            coords[index] = np.array([x,y,z])
            line = fp.readline()
            index += 1
    return coords

#------------------------------------------------------------------------------
# Try to load pre-calculated Odor matrixes
#------------------------------------------------------------------------------
if load_calculations and os.path.isfile(os.path.join(calculationsSavingPath,"connMatrixTotalOdorOn.npy")):
    connMatrixTotalOdorOn = np.load(os.path.join(calculationsSavingPath,"connMatrixTotalOdorOn.npy"))
    connMatrixTotalOdorOff = np.load(os.path.join(calculationsSavingPath,"connMatrixTotalOdorOff.npy"))
    connMatrixTotalOdorDiff = np.load(os.path.join(calculationsSavingPath,"connMatrixTotalOdorDiff.npy"))
    labels = np.load(os.path.join(calculationsSavingPath,"OdorLabels.npy"))
    nscouts = len(labels)
    print(f"Calculated Odor matrixes loaded from {calculationsSavingPath}!")
else:
    #------------------------------------------------------------------------------
    # Load the Data, Labels and define variables for Odor trials analysis
    #------------------------------------------------------------------------------
    print(f"Calculating Odor Matrixes")
    srate = 1000
    ntime   = 30*srate

    path = channelOdorPath

    labels = sio.loadmat(os.path.join(path, 'labels.mat'))
    labels = labels['labels'].tolist()[0]
    labels = [label[0] for label in labels]
    nscouts = len(labels)


    subj = 0
    connMatrixTotalOdorOn = np.zeros((nscouts,nscouts,21))
    connMatrixTotalOdorOff = np.zeros((nscouts,nscouts,21))
    connMatrixTotalOdorDiff = np.zeros((nscouts,nscouts,21))

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

    #---- Save the total matrixes in files ------
    if save_calculations:
        np.save(os.path.join(calculationsSavingPath,"connMatrixTotalOdorOn.npy"), connMatrixTotalOdorOn)
        np.save(os.path.join(calculationsSavingPath,"connMatrixTotalOdorOff.npy"), connMatrixTotalOdorOff)
        np.save(os.path.join(calculationsSavingPath,"connMatrixTotalOdorDiff.npy"), connMatrixTotalOdorDiff)
        np.save(os.path.join(calculationsSavingPath,"OdorLabels.npy"), labels)
        print(f"Calculated Odor matrixes saved in {calculationsSavingPath}!")


#------------------------------------------------------------------------------
# Try to load pre-calculated Placebo matrixes
#------------------------------------------------------------------------------
if load_calculations and os.path.isfile(os.path.join(calculationsSavingPath,"connMatrixTotalPlaceboOn.npy")):
    connMatrixTotalPlaceboOn = np.load(os.path.join(calculationsSavingPath,"connMatrixTotalPlaceboOn.npy"))
    connMatrixTotalPlaceboOff = np.load(os.path.join(calculationsSavingPath,"connMatrixTotalPlaceboOff.npy"))
    connMatrixTotalPlaceboDiff = np.load(os.path.join(calculationsSavingPath,"connMatrixTotalPlaceboDiff.npy"))
    labels = np.load(os.path.join(calculationsSavingPath,"PlaceboLabels.npy"))
    nscouts = len(labels)
    print(f"Calculated Placebo matrixes loaded from {calculationsSavingPath}!")
else:
    #------------------------------------------------------------------------------
    # Load the Data, Labels and define variables for Placebo trials analysis
    #------------------------------------------------------------------------------
    print(f"Calculating Placebo Matrixes")

    srate = 1000
    ntime   = 30*srate

    path = channelPlaceboPath

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

    #---- Save the total matrixes in files ------
    if save_calculations:
        np.save(os.path.join(calculationsSavingPath,"connMatrixTotalPlaceboOn.npy"), connMatrixTotalPlaceboOn)
        np.save(os.path.join(calculationsSavingPath,"connMatrixTotalPlaceboOff.npy"), connMatrixTotalPlaceboOff)
        np.save(os.path.join(calculationsSavingPath,"connMatrixTotalPlaceboDiff.npy"), connMatrixTotalPlaceboDiff)
        np.save(os.path.join(calculationsSavingPath,"PlaceboLabels.npy"), labels)
        print(f"Calculated Placebo matrixes saved in {calculationsSavingPath}!")


#------------------------------------------------------------------------------
# Mean across subjects and Figures creation
#------------------------------------------------------------------------------
print("Calculating means")
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
print("Doing Statistical analysis")
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

#---- Show plotted matrixes ------
show()

#------------------------------------------------------------------------------
# Read channel locations
#------------------------------------------------------------------------------
# coords = read_channLoc(channelLocationsPath)
coords = read_channLocMNI(channelLocationsPath)

#------------------------------------------------------------------------------
# Show visualization in browser
#------------------------------------------------------------------------------
view = view_connectome(significantconnectDiff, coords, edge_threshold='95%')
#view = view_connectome(significantconnectOdorVsOff, coords, edge_threshold='90%') 
view.open_in_browser() 
x = input("Done...press to exit")