import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import uncertainties as unc
from scipy import stats
from lmfit.models import GaussianModel


def plot(*args, ax=None, **kwargs):
    '''
    Quick interface to the plt.errorbar function:
    
    Parameters:
        *args:
            x: optinal, if given array-like. If it is an array of ufloats xerrorbars are plotted
            y: array-like. If it is an array of ufloats xerrorbars are plotted
        
        ax: optional, axis on which to execute the plot
        
        **kwargs: passed to plt.errorbar, e.g. 'fmt'
        
    Returns: ErrorbarContainer
        
    '''
    if len(args) == 2:
        xs = args[0]
        ys = args[1]
    else:
        ys = args[0]
        xs = np.arange(len(ys))
    
    if type(xs[0]) == unc.core.Variable or type(xs[0])==unc.core.AffineScalarFunc:
        x_d = np.array([[x.n, x.s] for x in xs])
        x_val = x_d[:,0]
        x_err = x_d[:,1]
    else:
        x_val = xs
        x_err = None
    if type(ys[0]) == unc.core.Variable or type(ys[0])==unc.core.AffineScalarFunc:
        y_d = np.array([[y.n, y.s] for y in ys])
        y_val = y_d[:,0]
        y_err = y_d[:,1]
    else:
        y_val = ys
        y_err = None
    
    if ax is not None:
        return ax.errorbar(x_val, y_val, xerr=x_err, yerr=y_err, **kwargs)
    return plt.errorbar(x_val, y_val, xerr=x_err, yerr=y_err, **kwargs)



class ExtendedKDE():
    def __init__(self, xs, ignore_Dirac_deltas=True):
        x_d = np.array([[x.n, x.s] for x in xs])
        x_val = x_d[:,0]
        x_err = x_d[:,1]
        
        min_idx = np.argmin(x_val)
        max_idx = np.argmax(x_val)
        self.min_x = x_val[min_idx] - 2*x_err[min_idx]
        self.max_x = x_val[max_idx] + 2*x_err[max_idx]
        self.gaussians = None
        
        if ignore_Dirac_deltas:
            self.gaussians = [stats.norm(*p) for p in x_d if p[1] > 0]
        
        else:
            self.gaussians = [stats.norm(*p) for p in x_d]
        
    def __call__(self, point):
        return np.sum([g.pdf(point) for g in self.gaussians])
    
    def evaluate(self, points):
        '''
        Performs a kde histogram of ufloat data

        Parameters:
            points: int or array-like:
                if int: number of points that are computed
                if array-like: array of points where to sample
        '''
        return points, np.array([self(point) for point in tqdm(points)])
    

    def plot(self, points=30, ax=None, xrange=None, switch_xy=False, **kwargs):
        '''
        Performs a kde histogram of ufloat data

        Parameters:
            points: int or array-like:
                if int: number of points that are computed
                if array-like: array of points where to sample
            ax: optional, axis on which to execute the plot
        '''
        if type(points) == int:
            if xrange is not None:
                points = np.linspace(*xrange, points)
            else:
                points = np.linspace(self.min_x, self.max_x, points)
            
        points, values = self.evaluate(points)
        
        if ax is not None:
            if switch_xy==False:
                ax.plot(points, values, **kwargs)
            else: 
                ax.plot(values, points, **kwargs)
        else:
            plt.plot(points,values, **kwargs)

        return points, values
    

    
def side_hist_plot(xdata, ydata, bins=30, external_axes=None, **kwargs):
    '''
    Makes a plot of 'ydata' vs 'xdata' with a kde plot of 'ydata' on its right.
    
    Params:
        'xdata': array-like
        'ydata': array-like of ufloats
        'bins': number of points for the kde plot
        'external_axes': tupple of size 2, external axes on which to do the plot
        
        **kwargs:
            figsize
            xlabel
            ylabel
            title
            
            label: label to create a legend in the plot
            fit_color: color of the fit line
            fit_linestyle
            
            adjust_ylims: bool. If True plot and histogram will have the same ylims
            
    Returns:
        fig: plt.figure
        axes: (ax_plot, ax_hist)
        fit_params: array of ufloats: [center, sigma] of the gaussian used for the fit.
    '''
    
    figsize = kwargs.pop('figsize', (10,7))
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    title = kwargs.pop('title', None)
    
    label = kwargs.pop('label', None)
    fit_color = kwargs.pop('fit_color', 'red')
    fit_linestyle = kwargs.pop('fit_linestyle', None)
    
    adjust_ylims = kwargs.pop('adjust_ylims', False)
    
    
    
    fig = None
    ax_plot = None
    ax_hist = None
    
    if external_axes is None:   
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4,5)

        ax_plot = fig.add_subplot(gs[:,0:3])
        ax_hist = fig.add_subplot(gs[:,3:])
    
    else:
        ax_plot, ax_hist = external_axes
        fig = ax_plot.figure
    
    # plot
    plot(xdata, ydata, ax=ax_plot, label=label, **kwargs)
    if label is not None:
        ax_plot.legend()
    if xlabel is not None:
        ax_plot.set_xlabel(xlabel)
    if ylabel is not None:
        ax_plot.set_ylabel(ylabel)
    if title is not None:
        ax_plot.set_title(title)
    if adjust_ylims:
        ax_hist.set_ylim(*ax_plot.get_ylim())
        
    # hist
    kernel = ExtendedKDE(ydata)
    
    x1,f1 = kernel.plot(points=bins, xrange=ax_plot.get_ylim(), ax=ax_hist, switch_xy=True, **kwargs)
    
    plt.setp(ax_hist.get_yticklabels(), visible=False) #Turn off tick labels
    
    # gaussian fit
    mod1 = GaussianModel(prefix='g1_')
    pars1 = mod1.guess(f1, x=x1)
    out1 = mod1.fit(f1, pars1, x=x1)
    ax_hist.plot(out1.best_fit, x1, color=fit_color, linestyle=fit_linestyle)
    
    # get fit parameters
    dist1 = unc.ufloat(out1.params['g1_center'].value, out1.params['g1_center'].stderr)
    sigma_dist1 = unc.ufloat(out1.params['g1_sigma'].value, out1.params['g1_sigma'].stderr)
    
    fit_params = np.array([dist1, sigma_dist1])
    
    return fig, (ax_plot, ax_hist), fit_params