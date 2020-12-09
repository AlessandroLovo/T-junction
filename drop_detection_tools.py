import numpy as np
import csv
import pandas as pd
#from PIL import Image
import matplotlib.pyplot as plt
#import os
from tqdm import tqdm
from lmfit.models import GaussianModel



# READ CSV FILE OUTPUTTED BY LABVIEW
def read_LV(folder, filename, plot_switch=True):
    '''
    Reads a csv file outputted by LabView
    
    Returns:
        sig1, sig2, t
    '''
    sig1=[]
    sig2=[]
    t = []
    
    folder.rstrip('/') + '/'
    
    # Data acquisition 
    with open(folder + filename) as data:
        for sig in csv.reader(data, delimiter='	'):
            #print(sig)
            sig[0] = sig[0].replace('.','').replace(',','.')
            sig[1] = sig[1].replace('.','').replace(',','.')
            sig[2] = sig[2].replace('.','').replace(',','.')
            #print(sig)
            sig1.append(float(sig[0]))
            sig2.append(float(sig[1]))
            t.append(float(sig[2]))
    sig1 = np.array(sig1)
    sig1 = sig1[0:-1]
    sig2 = np.array(sig2)
    sig2 = sig2[0:-1]
    t = np.array(t)
    t = t[0:-1]        
    
    if plot_switch:
        # Plot signal 1
        plt.figure()
        plt.figure(figsize=(20,4))
        plt.title('Signal 1')
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.plot(t, sig1, color='blue')
        plt.show()
        
        # Plot signal 2
        plt.figure()
        plt.figure(figsize=(20,4))
        plt.title('Signal 2')
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.plot(t, sig2, color='green')
        plt.show()
    
    return sig1, sig2, t

# The function does the Fast Fourier Transformation (FFT) and filters the signal keeping the frequency in [min_freq; max_freq]. 
# Then the function returns the anti-transformed filtered signal.
def FFT_cropping(signal, min_freq=1, max_freq=None, plot_switch=True):
    '''
    Makes the fft, crops it between 'min_freq' and 'max_freq' and then returns the ifft.
    If they are left None the FFT i not cropped
    '''
    
    max_p_freq = (len(signal) + 1)//2
    
    c0 = 0. # continuous component
    if min_freq == 0:
        c0 = np.mean(signal)
        min_freq = 1
    
    if max_freq is None or max_freq > max_p_freq:
        max_freq = max_p_freq
    # FFT of signal 
    F_sig = np.fft.fft(signal)                        

    if plot_switch:
        # FFT signal plot
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.set_xlabel('Frequency')
        axes.plot(F_sig[:max_p_freq], label='FFT')
        axes.set_ylim(0,200)

    # Signal filtering
    F_sig_crop       = np.zeros(len(F_sig))
    F_sig[:min_freq] = F_sig_crop[:min_freq]     # Set to 0 F_sig below min_freq
    F_sig[max_freq:] = F_sig_crop[max_freq:]     # Set to 0 F_sig above max_freq

    if plot_switch:
        # FFT signal filtered plot
        axes.plot(F_sig[:max_p_freq], label='FFT cropped')
        axes.legend()

    # Anti-FFT of F_sig
    sig_high = 2*np.fft.ifft(F_sig).real + c0
    
    return sig_high


# THRESHOLDS SEARCHING FUNCTION :::::::::::::::::::::::::::::::
def thr_searcher(Ydata, nbins=20, low_sigmas=3, high_sigmas=5, plot_switch=True, Xdata=None, ymin=None, ymax=None, c01=None, c02=None, **kwargs):
    
    '''
    Description:
        Oscillating data finds the two threshold to identify peaks (droplets)
    
    Params:
        - Ydata:       array with oscillating data
        - nbins:       number of bins for the histogram to find the two thresholds
        - low_sigmas:  number of sigmas above the low mean where to put the lower thr
        - high_sigmas: number of sigmas below the high mean where to put the higher thr
        - plot_switch: if True shows plots
        - Xdata:       array with the matching time (or space) to the data
        - ymin, ymax:  ylims for the plot
        
    Returns:
        thr_low, thr_high
    '''
    
    xlabel = kwargs.pop('xlabel',None)
    ylabel = kwargs.pop('ylabel',None)
    
    if xlabel is None:
        xlabel = 'position [mm]'
    if ylabel is None:
        ylabel = 'luminosity'
    
    if plot_switch:
        #plt.clf() # Clear figure
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        freq,bins,p = axes[0].hist(Ydata, nbins, **kwargs)
    
    # Histogram definition
    else:
        freq,bins,p = plt.hist(Ydata, nbins, color='green')
    x = 0.5 *(bins[:-1] + bins[1:])
    max_freq = np.max(freq)                                     
        
    # Gaussian 1
    gauss1 = GaussianModel(prefix='g1_')
    pars   = gauss1.make_params(center=x[0]+(x[-1]-x[0])/10, sigma=(x[-1]-x[0])/15 , amplitude=max(freq))

    # Gaussian 2
    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    
    if c01 is None:
        # Gaussian 1 parameters
        pars['g1_center'].set(max = x[0]+(x[-1]-x[0])*0.7, min=min(x))
        #pars['g1_sigma'].set(max=(x[-1]-x[0])/10, min=(x[-1]-x[0])/30)
        #pars['g1_amplitude'].set(value=max(freq)/20,min=10)
    else:
        pars['g1_center'].set(value = c01)

    # Gaussian 2 parameters
    if c02 is None:
        c02 = x[-1]-(x[-1]-x[0])/10
        
    pars['g2_center'].set(value=c02)
    pars['g2_sigma'].set(value=(x[-1]-x[0])/15)
    pars['g2_amplitude'].set(value=max(freq))

    mod  = gauss1 + gauss2
    init = mod.eval(pars, x=x)
    out  = mod.fit(freq, pars, x=x)

    
    if plot_switch:
        # Plot histo
        axes[0].plot(x, out.init_fit, 'k--', label='initial fit')
        axes[0].plot(x, out.best_fit, 'r-', label='best fit')
        axes[0].legend(loc='best')
        axes[0].set_title("Signal histogram")
        axes[0].set_xlabel(ylabel)
        axes[0].set_ylabel("Number of events")
        
        if axes[0].get_ylim()[1] > 1.5*max_freq:
            axes[0].set_ylim(0,1.5*max_freq)

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
    thr_low  = x_max1 + low_sigmas *sigma1
    thr_high = x_max2 - high_sigmas*sigma2
    
    # Signal plot
    if plot_switch:
        axes[0].vlines([thr_low], *axes[0].get_ylim(), color='cyan')
        axes[0].vlines([thr_high], *axes[0].get_ylim(), color='yellow')
        
        if Xdata is None:
            Xdata = np.arange(len(Ydata))
        
        axes[1].plot(Xdata, Ydata, **kwargs)
        axes[1].plot(thr_high*np.ones(len(Xdata)), color='yellow', label='thr_high')
        axes[1].plot(thr_low *np.ones(len(Xdata)), color='cyan', label='thr_low')
        plt.legend()
        plt.xlim((0, Xdata[len(Xdata)-1]))
        if not (ymin is None or ymax is None):
            axes[1].set_ylim(ymin, ymax)
        axes[1].set_title("Signal with thresholds")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        
    if thr_low > thr_high:
        print('WARNING: thr_low > thr_high')
    
    return thr_low, thr_high


# DROP DETECTION FUNCTION :::::::::::::::::::::::::::::::::::::::::::::::::::
def drop_det(Xdata, Ydata, thr_low, thr_high, plot_switch=True, ymin=None, ymax=None, xrange=None, **kwargs):
    
    '''
    Description:
        Identifies the start and end position of the droplets
    
    Params:
        - Xdata:             array with time (space)
        - Ydata:             array with the voltage/luminosity values
        - thr_low, thr_high: threshold computed with 'thr_seracher'
        - plot_switch:       if True shows plots
        - ymin, ymax:        ylims for the plot
        - xrange:            width of the window to be shown (expressed in time/space units)
        
    Returns:
        drop_start, drop_end: arrays with starts and ends of the droplets
                              They have always the same lenght and drop_start[0] < drop_end[0],
                              i.e. no spurious detections
    '''
    
    # Drops edges computing
    bool_high  = Ydata > thr_high
    bool_low   = Ydata < thr_low
    drop_start = [0]
    drop_end   = [1]
   
    # Detection
    for i in range(len(Ydata)-1):
        
        if bool_high[i]==False and bool_low[i+1]==False and bool_high[i+1]==True:
            if drop_start[-1] < drop_end[-1] and i > drop_end[-1]:
                drop_start.append(i)
                
        elif bool_low[i]==False and bool_low[i+1]==True and bool_high[i+1]==False:
            if drop_start[-1] > drop_end[-1] and i > drop_start[-1]:
                drop_end.append(i)
 
    # Number acquisition -> time [s] conversion
    drop_start = Xdata[drop_start]
    drop_end   = Xdata[drop_end]
    
    # Selection
    drop_start = drop_start[1:]
    drop_end   = drop_end[1:]
    
    # Cropping
    if len(drop_start) > len(drop_end):
        drop_start = drop_start[:-1]
    
    # Plotting 
    if plot_switch:
        if xrange is None:
            xrange = Xdata[-1]
        for j in range(int (Xdata[-1]/xrange)):
            fig, ax = plt.subplots(figsize=(20,4))
            plt.plot(Xdata, Ydata, **kwargs)
            
            if ymin is None or ymax is None:
                ymin, ymax = ax.get_ylim()
            else:
                plt.ylim(ymin, ymax)
            
            for i in range(len(drop_end)):

                plt.vlines(drop_start[i], ymin, ymax, color='green')
                plt.vlines(drop_end[i],   ymin, ymax, color='red')

            plt.ylabel("Luminosity")
            plt.xlabel("Position [mm]")
            plt.xlim(j*xrange,(j+1)*xrange)
            plt.plot(thr_high*np.ones(len(Xdata)), color='yellow')
            plt.plot(thr_low *np.ones(len(Xdata)), color='cyan')
            plt.show()
        
    return np.array(drop_start), np.array(drop_end)