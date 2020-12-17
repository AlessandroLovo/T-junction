import numpy as np
import csv
import pandas as pd
#from PIL import Image
import matplotlib.pyplot as plt
#import os
from tqdm import tqdm
from lmfit.models import GaussianModel
from scipy import optimize as optim



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
    
    # Signals plots
    if plot_switch:
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15,6))
        axs[0].plot(t, sig1, color='blue', label= "signal 1")
        axs[0].set_ylabel('Voltage [V]')
        axs[1].plot(t, sig2, color='green', label= "signal 2")
        axs[1].set_ylabel('Voltage [V]')
        axs[1].set_xlabel('Time [s]')
        fig.legend(loc='best')
    
    return sig1, sig2, t



# RESAMPLNIG OF THE SIGNAL
def resample(sig, t, dt):
    
    '''
    Resample every dt a signal 'sig' that is sampled at each element of array 't'
    '''
    new_t = np.arange(t[0], t[-1], dt)
    
    new_sig = []
    t_index = 0
    t0 = t[t_index]
    t1 = t[t_index + 1]
    x0 = sig[t_index]
    x1 = sig[t_index + 1]
    
    
    for p in new_t:
        while(p > t1):
            t_index += 1
            t0 = t[t_index]
            t1 = t[t_index + 1]
            x0 = sig[t_index]
            x1 = sig[t_index + 1]
            
        new_sig.append((x0*(t1 - p) + x1*(p - t0))/(t1 - t0))
        
    new_sig = np.array(new_sig)
    
    return new_sig, new_t


# RECTIFY FUNCTION 
def rectify_new(signal, xrange, xdata=None, ignore_bias=-1, manual_thr=-np.inf, plot_switch=True, **kwargs):
    '''
    Uses a running mean to straighten a signal
    
    Parameters:
        'signal'
        'xrange': width of the running mean window in 'xdata' units
        'xdata': if None is np.arange(len(signal))
        'ignore_bias': half width of the horizontal region centered on the pivot where data are ignored in the computation of the mean
        'manual_thr': all data below it are ignored in the computation of the mean
        
        'plot_switch': toggle plots
        
        '**kwargs':
            xmin, xmax: manual xlim for the plots
            xlabel
            ylabel
            
    '''
    
    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)
    xlabel = kwargs.pop('xlabel', 'Time [s]')
    ylabel = kwargs.pop('ylabel', 'Voltage [V]')
    
    # Y-data
    signal = np.array(signal)
    
    # X-data
    if xdata is None:
        xdata = np.arange(len(signal))
    else:
        xdata = np.array(xdata)
    
    # Analysis
    main_mean  = np.mean(signal)
    lower_mean = np.mean([s for s in signal if s < main_mean])
    upper_mean = np.mean([s for s in signal if s >= main_mean])
    pivot      = 0.5*(lower_mean + upper_mean)
    
    # Fit
    def f_upper(x):
        arr = np.where((xdata < x + xrange) * (xdata > x - xrange))
        max_len = len(arr)
        arr = signal[arr]
        arr = arr[arr > pivot + ignore_bias]
        arr = np.concatenate([arr, np.array([upper_mean]*(max_len - len(arr)))])
        return np.mean(arr)

    
    def f_lower(x):
        arr = np.where((xdata < x + xrange) * (xdata > x - xrange))
        max_len = len(arr)
        arr = signal[arr]
        arr = arr[arr < pivot - ignore_bias]
        arr = arr[arr > manual_thr]
        arr = np.concatenate([arr, np.array([lower_mean]*(max_len - len(arr)))])
        return np.mean(arr)
    
    fit_curve_upper = np.array([f_upper(x) for x in xdata])
    fit_curve_lower = np.array([f_lower(x) for x in xdata])
    
    new_sig = np.copy(signal)
    for i,s in enumerate(new_sig):
        if s > pivot:
            new_sig[i] += upper_mean - fit_curve_upper[i]
        else:
            new_sig[i] += lower_mean - fit_curve_lower[i]

   

    # Plots ---------------------------------------------------
    fig = None
    if plot_switch:
        
        y_name = 'y'
        if ylabel is not None:
            y_name = ylabel[0]
        
        # Labels
        main_mean_label  = "$\overline\{\r%s\}$" %y_name
        lower_mean_label = "$\overline{\r%s}_{down}$" %y_name
        upper_mean_label = "$\overline{\r%s}_{up}$" %y_name
        pivot_label      = "$\overline{\r%s}_{pivot}$" %y_name
        manual_thr_label = "$%s_{thr}$" %y_name
        thr_label        = "$\overline{\r%s}_{pivot} \pm %s_{bias}$" %(y_name, y_name)
        fit_up_label     = "fit curve (upper)"
        fit_low_label    = "fit curve (lower)"
        
        # Thresholds plot
        fig,axs = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
        axs[0].plot(xdata, signal, color='green')
        axs[0].plot(xdata, upper_mean*np.ones(len(xdata)),          'y-',  label=upper_mean_label)
        axs[0].plot(xdata, lower_mean*np.ones(len(xdata)),          'c-',  label=lower_mean_label)
        axs[0].plot(xdata, manual_thr*np.ones(len(xdata)),          'b-',  label=manual_thr_label)
        axs[0].plot(xdata, pivot*np.ones(len(xdata)),               'k-',  label=pivot_label)
        axs[0].plot(xdata, (pivot+ignore_bias)*np.ones(len(xdata)), 'k--', label=thr_label)
        axs[0].plot(xdata, (pivot-ignore_bias)*np.ones(len(xdata)), 'k--')
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel) 
        if not (xmin is None or xmax is None):
                axs[0].set_xlim(xmin, xmax)
        axs[0].legend()
    
        # Fit plot 
        axs[1].plot(xdata, signal, 'g-')
        axs[1].plot(xdata, fit_curve_upper, 'y-', label=fit_up_label)
        axs[1].plot(xdata, fit_curve_lower, 'c-', label=fit_low_label)
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel) 
        if not (xmin is None or xmax is None):
                axs[1].set_xlim(xmin, xmax)
        axs[1].legend()
    
        # Final signal plot
        fig,axs = plt.subplots(nrows=1, ncols=1, figsize=(15,6))
        axs.plot(xdata, signal,  'g-', label = "original signal")
        axs.plot(xdata, new_sig, 'r-', label = "rectified signal")
        axs.set_xlabel(xlabel)
        axs.set_ylabel(ylabel) 
        if not (xmin is None or xmax is None):
                axs.set_xlim(xmin, xmax)
        fig.legend()
    
    return fig, new_sig



# FFT FILTERING 
def FFT_cropping(signal, Xdata=None, min_freq=1, max_freq=None, plot_switch=True):
   
    '''
    Makes the fft, crops it between 'min_freq' and 'max_freq' and then returns the ifft.
    If they are left None the FFT iS not cropped
    '''
    
    max_p_freq = (len(signal) + 1)//2
    
    c0 = 0. # continuous component
    if min_freq == 0:
        c0 = np.mean(signal)
        min_freq = 1
    
    if max_freq is None or max_freq > max_p_freq:
        max_freq = max_p_freq
     
    xlabel = "Time [s]"
    if Xdata is None:
        Xdata  = np.arange(len(Ydata))
        xlabel = "Measure index"
        
    # FFT of signal 
    F_sig = np.fft.fft(signal)                        

    # Plots
    if plot_switch:
        # FFT signal plot
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
        axs[0].set_title("Signal frequency spectrum")
        axs[0].set_xlabel('Frequency')
        axs[0].plot(F_sig[:max_p_freq], color= "blue", label='FFT')
        axs[0].set_ylim(0,200)
        
    # Signal filtering
    F_sig_crop       = np.zeros(len(F_sig))
    F_sig[:min_freq] = F_sig_crop[:min_freq]     # Set to 0 F_sig below min_freq
    F_sig[max_freq:] = F_sig_crop[max_freq:]     # Set to 0 F_sig above max_freq
    
    # Anti-FFT of F_sig
    sig_high = 2*np.fft.ifft(F_sig).real + c0

    # Plots
    if plot_switch:
        axs[0].plot(F_sig[:max_p_freq], color= "orange", label='FFT cropped')
        axs[0].legend()
        # Signals plots
        axs[1].set_title("Signal")
        axs[1].plot(Xdata, signal,   color='blue', label= "original signal")
        axs[1].set_ylabel('Voltage [V]')
        axs[1].set_xlabel(xlabel)
        axs2 = axs[1].twinx() 
        axs2.tick_params(axis = 'y', labelcolor = "red")
        axs2.plot(Xdata, sig_high, color='red',  label= "cropped signal")
        axs2.set_ylabel('Voltage [V]')
        legend = fig.legend(['original signal','cropped signal'], loc='best')
    
    return sig_high


# THRESHOLDS SEARCHING FUNCTION 
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
        xlabel = 'Position [mm]'
    if ylabel is None:
        ylabel = 'Luminosity'
    
    if plot_switch:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
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
        #axes[0].plot(x, out.init_fit, 'k--', label='initial fit')
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
    
    # Labels
    label_low  = "thr. low (" + str(low_sigmas) + "$\sigma$)"
    label_high = "thr. high (" + str(high_sigmas) + "$\sigma$)"
    
    # Signal plot
    if plot_switch:
        axes[0].vlines([thr_low], *axes[0].get_ylim(), color='cyan')
        axes[0].vlines([thr_high], *axes[0].get_ylim(), color='yellow')
        
        if Xdata is None:
            Xdata = np.arange(len(Ydata))
        
        axes[1].plot(Xdata, Ydata, **kwargs)
        axes[1].plot(thr_high*np.ones(len(Xdata)), color='yellow', label=label_low)
        axes[1].plot(thr_low *np.ones(len(Xdata)), color='cyan', label=label_high)
        plt.legend()
        plt.xlim((0, Xdata[len(Xdata)-1]))
        if not (ymin is None or ymax is None):
            axes[1].set_ylim(ymin, ymax)
        axes[1].set_title("Signal with thresholds")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        
    if thr_low > thr_high:
        print('WARNING: thr_low > thr_high')
    
    return thr_low, thr_high, sigma1, sigma2 # sigma1 is the lower one



# DROP DETECTION FUNCTION 
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



## new version
# print(bounds) #new drop detection: narrow_start, narrow_end, wide_start, wide end
def drop_det_new(Xdata, Ydata, thr_low, thr_high, backward_skip = 1, forward_skip = 1, return_indexes=True,
                 plot_switch=True, ymin=None, ymax=None, xrange=None, **kwargs):
    
    '''
    Description:
        Identifies the start and end position of the droplets in the narrow and wide range
    
    Params:
        - Xdata:             array with time (space)
        - Ydata:             array with the voltage/luminosity values
        - thr_low, thr_high: threshold computed with 'thr_seracher'
        - plot_switch:       if True shows plots
        - ymin, ymax:        ylims for the plot
        - xrange:            width of the window to be shown (expressed in time/space units)
        
    Returns:
        narrow_start, narrow_end, wide_start, wide_end : arrays with starts and ends of the droplets
                                                         They have always the same lenght and *_start[0] < *_end[0],
                                                         i.e. no spurious detections
    '''
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    
    
    # Masks
    bool_high  = Ydata > thr_high
    bool_low   = Ydata < thr_low
    
    #first cycle finds narrow_start. 
    #drop_ends are detected when drop goes below thr_low.
    #we don't need them, but they are necessary to the correct narrow_start detection
    #narrow_start is what we previously called drop_start, drop_end is the old drop_end
    narrow_start = [0]
    drop_end   = [1]
   
    for i in range(len(Ydata)-1):
        
        if bool_high[i]==False and bool_low[i+1]==False and bool_high[i+1]==True:
            if narrow_start[-1] < drop_end[-1] and i > drop_end[-1]:
                narrow_start.append(i)
                
        elif bool_low[i]==False and bool_low[i+1]==True and bool_high[i+1]==False:
            if narrow_start[-1] > drop_end[-1] and i > narrow_start[-1]:
                drop_end.append(i)
                
    #second cycle finds narrow_end. Now we call ascent_start the point where the drop goes beyond thr_low.
    ascent_start=[0]
    narrow_end=[1]
    
    for i in range(len(Ydata)-1):
        
        if bool_low[i]==True and bool_high[i]==False and bool_low[i+1]==False:
            if ascent_start[-1] < narrow_end[-1] and i > narrow_end[-1]:
                ascent_start.append(i)
                
        elif bool_low[i]==False and bool_high[i]==True and bool_high[i+1]==False  :
            if i > ascent_start[-1]:
                if ascent_start[-1] > narrow_end[-1]:
                    narrow_end.append(i)
                else: 
                    narrow_end[-1]=i      #make sure to take the real narrow_end, not just a fluctuation in the middle of the drop
                
                
    # Selection
    narrow_start = narrow_start[1:]
    narrow_end   = narrow_end[1:]
    ascent_start = ascent_start[1:]
    drop_end     = drop_end[1:] 
    # Cropping
    if len(narrow_start) > len(narrow_end):
        narrow_start = narrow_start[:-1] 
    print(len(narrow_start), len(narrow_end))
                
    #find wide_start and wide_end. Only need low thr:
    #check whenever signal simply overcomes low threshold and stores those indices
    # wide start: last spike that goes beyond low threshold before ascent_start
    # wide_end: first spike to go below low threshold after drop_end
    
    spike_start=[]
    spike_end=[]
    wide_start=[]
    wide_end=[]
    
    for i in range(len(Ydata)-1):
        if bool_low[i]==True and bool_low[i+1]==False:
            spike_start.append(i)
        elif bool_low[i]==False and bool_low[i+1]==True:
            spike_end.append(i)
            
    spike_start = np.array(spike_start)
    spike_end = np.array(spike_end)
    
    b=0
    a=0
    for start,end in list(zip(narrow_start,narrow_end)):
        #print(start,end)
        if len(spike_start[spike_start<start])>1:
            if backward_skip > 0:
                a_start = spike_start[spike_start<start][-(1 + backward_skip)]
                a_end = spike_start[spike_start<start][-(backward_skip)] - 1
                peak_idx = np.argmax(Ydata[a_start:a_end])
                a = a_start + peak_idx
            else:
                a = spike_start[spike_start<start][-(1 + backward_skip)]
            wide_start.append(a)
        if b>a: print(Xdata[start],'s: WRONG WIDE DROP DETECTION')
        if len(spike_end[spike_end>end])>1:
            if forward_skip > 0:
                b_start = spike_end[spike_end>end][forward_skip - 1] + 1
                b_end = spike_end[spike_end>end][forward_skip]
                peak_idx = np.argmax(Ydata[b_start:b_end])
                b = b_start + peak_idx
            else:
                b = spike_end[spike_end>end][forward_skip]
            wide_end.append(b)
        
    #cropping
    if len(wide_start) > len(wide_end):
        wide_start = wide_start[:-1] 
    if len(narrow_start)>len(wide_start):
        narrow_start = narrow_start[:-1] 
        narrow_end = narrow_end[:-1] 
        
    print(len(narrow_start), len(narrow_end))
    
    # Plotting 
    if plot_switch:
        if xrange is None:
            xrange = Xdata[-1]
        for j in range(int (Xdata[-1]/xrange)):
            fig, ax = plt.subplots(figsize=(15,6))
            plt.plot(Xdata, Ydata, **kwargs)
            
            if ymin is None or ymax is None:
                ymin, ymax = ax.get_ylim()
            else:
                plt.ylim(ymin, ymax)
            
            for i in range(len(narrow_end)-1):

                plt.vlines(Xdata[narrow_start[i]], ymin, ymax, color='green',  label="start (narrow)")
                plt.vlines(Xdata[narrow_end[i]],   ymin, ymax, color='red',    label="end (narrow)")
                plt.vlines(Xdata[wide_start[i]],   ymin, ymax, color='lime',   label="start (wide)")
                plt.vlines(Xdata[wide_end[i]],     ymin, ymax, color='orange', label="end (wide)")

            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            plt.xlim(j*xrange,(j+1)*xrange)
            plt.plot(thr_high*np.ones(len(Xdata)), color='yellow', label = "thr. high")
            plt.plot(thr_low *np.ones(len(Xdata)), color='cyan',   label = "thr. low")
            plt.show()

         
    if not return_indexes: #         Number acquisition -> time [s] conversion
        narrow_start = Xdata[narrow_start]
        narrow_end   = Xdata[narrow_end]
        wide_start   = Xdata[wide_start]
        wide_end     = Xdata[wide_end]
    
        
    return np.array(narrow_start), np.array(narrow_end), np.array(wide_start), np.array(wide_end)