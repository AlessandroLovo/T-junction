import numpy as np
#import pandas as pd
#from PIL import Image
import matplotlib.pyplot as plt
#import os
from tqdm import tqdm
from lmfit.models import GaussianModel


# THRESHOLDS SEARCHING FUNCTION :::::::::::::::::::::::::::::::
def thr_searcher(data, nbins=20, low_sigmas=3, high_sigmas=5, t=None, plot_switch=True, ymin=None, ymax=None):
    '''
    oscillating data finds the two threshold to identify peaks (droplets)
    
    params:
        data: array with oscillating data
        nbins: number of bins for the histogram to find the two thresholds
        low_sigmas: number of sigmas above the low mean where to put the lower thr
        high_sigmas: number of sigmas below the high mean where to put the higher thr
        
        t: array with the matching time (or space) to the data
        
        plot_switch: if True shows plots
        ymin, ymax: ylims for the plot
        
    Returns:
        thr_low, thr_high
    '''
    
    # Histogram definition
    freq,bins,p = plt.hist(data, nbins, color='green')
    x = 0.5 *(bins[:-1] + bins[1:])
        

    # Gaussian 1
    gauss1 = GaussianModel(prefix='g1_')
    pars   = gauss1.make_params(center=x[0]+(x[-1]-x[0])/10, sigma=(x[-1]-x[0])/15 , amplitude=max(freq))

    # Gaussian 2
    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())

    # Gaussian 1 parameters
    pars['g1_center'].set(max = x[0]+(x[-1]-x[0])*0.7, min=min(x))
    #pars['g1_sigma'].set(max=(x[-1]-x[0])/10, min=(x[-1]-x[0])/30)
    #pars['g1_amplitude'].set(value=max(freq)/20,min=10)

    # Gaussian 2 parameters
    pars['g2_center'].set(value=x[-1]-(x[-1]-x[0])/10)
    pars['g2_sigma'].set(value=(x[-1]-x[0])/15)
    pars['g2_amplitude'].set(value=max(freq))

    mod  = gauss1 + gauss2
    init = mod.eval(pars, x=x)
    out  = mod.fit(freq, pars, x=x)

    
    if plot_switch:
        plt.clf() # Clear figure
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        
        # Plot histo
        axes[0].hist(data, nbins, color='green')
        axes[0].plot(x, out.init_fit, 'k--', label='initial fit')
        axes[0].plot(x, out.best_fit, 'r-', label='best fit')
        axes[0].legend(loc='best')
        axes[0].set_title("Signal histogram")
        axes[0].set_xlabel("Luminosity")
        axes[0].set_ylabel("Number of events") 

    center2 = out.best_values.get('g2_center')
    sigma2  = out.best_values.get('g2_sigma')
    center1 = out.best_values.get('g1_center')
    sigma1  = out.best_values.get('g1_sigma')

    
    max_freq1 = max(freq[x<center1+sigma1])
    x_max1    = x[np.argmax(freq[x<center1+2*sigma1])]
    max_freq2 = max(freq[x>center2-sigma2])
    a         = x > center2-sigma2
    freq2     = np.zeros(len(freq))
    for i in range(len(a)): 
        if a[i]==True: freq2[i]=freq[i]
    x_max2 = x[np.argmax(freq2)]
       
    # Thresholds computing
    thr_low  = x_max1 + low_sigmas*sigma1
    thr_high = x_max2 - high_sigmas*sigma2
    
    # Signal plot
    if plot_switch:
        if t is None:
            t = np.arange(len(data))
        
        axes[1].plot(t, data, color='green')
        axes[1].plot(thr_high*np.ones(len(t)), color='red')
        axes[1].plot(thr_low*np.ones(len(t)), color='red')
        plt.xlim((0,t[len(t)-1]))
        if not (ymin is None or ymax is None):
            axes[1].set_ylim(ymin, ymax)
        axes[1].set_title("Signal with thresholds")
        axes[1].set_xlabel("Position [mm]")
        axes[1].set_ylabel("Luminosity") 
    
    return thr_low, thr_high


# DROP DETECTION FUNCTION :::::::::::::::::::::::::::::::::::::::::::::::::::
def drop_det(time, data, thr_low, thr_high, plot_switch=True, ymin=None, ymax=None, xrange=None):
    '''
    Identifies the start and end position of the droplets
    
    params:
        time: array with time (space)
        data: array with the voltage/luminosity values
        thr_low, thr_high: threshold computed with 'thr_seracher'
        
        plot_switch: if True shows plots
        ymin, ymax: ylims for the plot
        xrange: width of the window to be shown (expressed in time/space units)
        
    returns:
        drop_start, drop_end: arrays with starts and ends of the droplets
                            They have always the same lenght and drop_start[0] < drop_end[0],
                            i.e. no spurious detections
    '''
    
    # Drops edges computing
    bool_high  = data > thr_high
    bool_low   = data < thr_low
    drop_start = [0]
    drop_end   = [1]
   
    # Detection
    for i in range(len(data)-1):
        
        if bool_high[i]==False and bool_low[i+1]==False and bool_high[i+1]==True:
            if drop_start[-1] < drop_end[-1] and i > drop_end[-1]:
                drop_start.append(i)
                
        elif bool_low[i]==False and bool_low[i+1]==True and bool_high[i+1]==False:
            if drop_start[-1] > drop_end[-1] and i > drop_start[-1]:
                drop_end.append(i)
 
    # Number acquisition -> time [s] conversion
    drop_start = time[drop_start]
    drop_end   = time[drop_end]
    
    # Selection
    drop_start = drop_start[1:]
    drop_end   = drop_end[1:]
    
    # Cropping
    if len(drop_start) > len(drop_end):
        drop_start = drop_start[:-1]
    
    # Plotting 
    if plot_switch:
        if xrange is None:
            xrange = time[-1]
        for j in range(int (time[-1]/xrange)):
            fig, ax = plt.subplots(figsize=(20,4))
            plt.plot(time, data)
            
            if ymin is None or ymax is None:
                ymin, ymax = ax.get_ylim()
            else:
                plt.ylim(ymin, ymax)
            
            for i in range(len(drop_end)):

                plt.vlines(drop_start[i], ymin, ymax, color='green')
                plt.vlines(drop_end[i], ymin, ymax, color='red')

            plt.ylabel("Luminosity")
            plt.xlabel("Position [mm]")
            plt.xlim(j*xrange,(j+1)*xrange)
            
            plt.plot(thr_high*np.ones(len(s)), color='yellow')
            plt.plot(thr_low*np.ones(len(s)), color='yellow')
            plt.show()
        
    return drop_start, drop_end