#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 19:42:24 2020

@author: alessandro
"""

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm


def extract_frames(path,video_name):
    '''
    Splits the video in its frames and moves video and frame to a folder named as the video
    '''
    path = path.rstrip('/')
    fol = path + '/' + video_name.rsplit('.',1)[0]
    
    if os.path.exists(fol):
        print('Frames already extracted :)')
        return fol
    if not os.path.exists(path + '/' + video_name):
        raise FileNotFoundError('No such file or directory')
    
    os.mkdir(fol)
    cur_dir = os.getcwd()
    os.replace(path + '/' + video_name, fol + '/' + video_name)
    
    os.chdir(fol)
    os.system(f'ffmpeg -i {video_name} frames_%04d.jpg')
    
    os.chdir(cur_dir)
    return fol
    
def get_arrays(folder,color=1,max_frames=None, start_frame_idx=0):
    '''
    Read the frames in 'folder' and compute their mean.
    For memory reasons only one of the r,g,b channels needs to be selected with the variable 'color': respectively 0,1,2
    'max_frames' is the maximum number of frames to analyze, by default all in the directory
    
    Returns:
        arrays: array of frames as 2d arrays
        mean_array: mean of all the images as a 2d array
    '''
    folder = folder.rstrip('/')
    images = []
    names = np.sort(os.listdir(folder))
    i = 0
    if max_frames is None:
        max_frames = len(names)
    for name in names:
        if name.startswith('frames'):
            frame_idx = name.split('.')[0].split('_')[1]
            if int(frame_idx) >= start_frame_idx:
                images.append(Image.open(folder + '/' + name))
                i += 1
        if i >= max_frames:
            break
    print(f'Last frame analyzed: {frame_idx}')
            
    arrays = np.array([np.asarray(image)[:,:,color] for image in images])
    
    mean_array = np.zeros_like(arrays[0],dtype=float)
    for array in arrays:
        mean_array += array
    mean_array /= (arrays.shape[0])
    
    mean_array = np.array(mean_array,dtype=np.uint8)
    
    return arrays, mean_array
    
    
def extend(array, new_shape=(960,1600)):
    '''
    Extend a gray scale image into a bigger one
    '''
    new_array = np.zeros(new_shape,dtype=np.uint8)
    offset_x = (new_shape[0] - array.shape[0])//2
    offset_y = (new_shape[1] - array.shape[1])//2
    
    new_array[offset_x:(offset_x + array.shape[0]), offset_y:(offset_y + array.shape[1])] = array
    return new_array

def subtract_mean(arrays, mean_array, negative=False, batch_size=100, verbose=False):
    '''
    Smart subtraction of the mean to avoid overflow
    '''
    diffs = []
    m = 255
    M = -255
    for i in range(len(arrays)//batch_size + 1):
        if verbose:
            print(f'batch {i}')
        partial = np.array([array*1. - mean_array for array in arrays[i*batch_size : (i + 1)*batch_size]])
        if len(partial):
            m = min(m, np.min(partial))
            M = max(M, np.max(partial))
            diffs.append(partial)
    
    print(f'{m = }, {M = }')
    for i in range(len(diffs)):
        diffs[i] -= m
    
    if M - m > 255:
        print('Subtracting the bias will generate some overflow')    
        o = input('Rescale arrays to avoid overflow? [y/n] ')
        if o == 'y':
            for i in range(len(diffs)):
                diffs[i] *= (255/(M - m))
    
    if verbose:
        print('converting to np.uint8')
    for i in range(len(diffs)):
        diffs[i] = np.array(diffs[i], dtype=np.uint8)
    
    if negative:
        if verbose:
            print('making the negative')
        for i in range(len(diffs)):
            diffs[i] = 255 - diffs[i]
    
    if verbose:
        print('concatenating')
    
    return np.concatenate(diffs)
    

def preprocess(array_sub,rotation=35,filter_size=0, new_shape=(960,1600)):
    '''
    Rotates the image by 'rotation' degrees and then applies a gaussain filter
    if 'filter_size' > 1
    
    if 'new_shape' == (0,0): the new_shape is automatically computed
    
    returns the preprocessed image
    '''
    
    if new_shape == (0,0):
        old_shape = array_sub.shape
        new_shape = (old_shape[0], int(np.sqrt(old_shape[0]**2 + old_shape[1]**2)))
    
    img = Image.fromarray(extend(array_sub,new_shape)).rotate(rotation)
    
    if filter_size != 0:
        img = Image.fromarray(ndimage.gaussian_filter(img,filter_size))
        
    return img
    


class Channel_analyzer():
    
    def __init__(self):
        
        # for isolating the channel
        self.points = []
        self.thickness = 10
        
        # variables for plotting
        self.points_on_ax = None
        self.highlight_on_ax = None
        self.u_border_on_ax = None
        self.l_border_on_ax = None
        self.shown_borders = False
        self.sorted_points = True
        
        self.current_index = -1
        
        # for calibrating the images
        self.real_distance = 0. # in mm
        self._c_constant = None
        self.c_points = []
        
        # variables for manual disance measuring
        self.d_points = []
        self.d_current_index = 0
        self.d_points_on_ax = None
        self.d_highlight_on_ax = None
        
        # variables for plotting
        self.c_points_on_ax = None
        self.c_highlight_on_ax = None
        self.c_current_index = -1
        
        # variables for analysis
        self._central_ys = None
        self._s = None
        
    def save(self,name,folder='./'):
        fol = folder.rstrip('/') + '/' + name.rstrip('/')
        if os.path.exists(fol):
            o = input(fol + ' exists: overwrite? [y/n]')
            if o != 'y':
                return
            print('Overwriting')
        else:
            os.mkdir(fol)
            
        np.save(fol+'/points.npy', self.points)
        np.save(fol+'/c_points.npy', self.c_points)
        np.save(fol+'/other.npy', [self.thickness,self.real_distance])
        
    def load(self,name,folder='./'):
        fol = folder.rstrip('/') + '/' + name.rstrip('/')
        if not os.path.exists(fol):
            print(fol + ' does not exist')
            return
        
        self.points = np.load(fol+'/points.npy').tolist()
        self.c_points = np.load(fol+'/c_points.npy').tolist()
        other = np.load(fol+'/other.npy')
        self.thickness = int(other[0])
        self.real_distance = other[1]
    
        
    @property
    def s(self):
        if self._s is None:
            self._s = []
            s_coord = 0.
            point_idx = 0
            x = self.points[0][0]
            dilation = 0.
            def compute_dilation():
                return np.sqrt((self.points[point_idx + 1][0] - self.points[point_idx][0])**2 + 
                                (self.points[point_idx + 1][1] - 
                                    self.points[point_idx][1])**2)/(self.points[point_idx + 1][0] -
                                                                    self.points[point_idx][0])
            dilation = compute_dilation()
            
            while(x < self.points[-1][0]):
                self._s.append(s_coord)
                s_coord += dilation
                if x == self.points[point_idx + 1][0]:
                    point_idx += 1
                    dilation = compute_dilation()
                x += 1
            # do the last point
            self._s.append(s_coord)
            
            self._s = np.array(self._s)
            self._s *= self.c_constant
            
        return self._s
    
    @property
    def central_ys(self):
        if self._central_ys is None:
            self._central_ys = []
            point_idx = 0
            x = self.points[0][0]
            def compute_inclination():
                return (self.points[point_idx + 1][1] - 
                        self.points[point_idx][1])/(self.points[point_idx + 1][0] - 
                                                    self.points[point_idx][0])
            inclination = compute_inclination()
            while(x < self.points[-1][0]):
                if x == self.points[point_idx + 1][0]:
                    point_idx += 1
                    inclination = compute_inclination()
                    y = self.points[point_idx][1]
                else:
                    y = self.points[point_idx][1] + int((x - self.points[point_idx][0])*inclination)
                self._central_ys.append(y)
                x += 1
            # do the last point
            y = self.points[-1][1]
            self._central_ys.append(y)
            
            self._central_ys = np.array(self._central_ys)
            
        return self._central_ys
            
            
    @property
    def c_constant(self): # calibration constant: from distance in pixels to distance in mm
        if self._c_constant is None:
            pixel_distance = np.sqrt((self.c_points[0][0] - self.c_points[1][0])**2 +
                                     (self.c_points[0][1] - self.c_points[1][1])**2)
            self._c_constant = self.real_distance/pixel_distance
        return self._c_constant
    
    
    def evaluate(self,img):
        '''
        Compute the behavior of the luminosity of the pixels along the channel
        
        Input:
            img: Image object
            
        Returns:
            s: array with the coordinate along the channel in mm
            mean: array with the mean luminosity
            std: array with the std of the luminosity
        '''
        array = np.asarray(img)
        
        mean = []
        std = []
        
        for i,y in enumerate(self.central_ys):
            x = self.points[0][0] + i
            a = array[y - self.thickness : y + self.thickness + 1, x] # first y then x
            mean.append(np.mean(a))
            std.append(np.std(a,ddof=1))
        
        mean = np.array(mean)
        std = np.array(std)
        
        return self.s, mean, std

        
    
    def calibrate(self,img,real_distance, **kwargs):
        '''
        Allows to calibrate the object to distances in mm
        
        Input:
            img: image with an object of known size
            real_distance: distance in mm between the two points you are going to put on the image
            
        Requirements:
            You need to have %matplotlib notebook
        
        Controls:
            Right click to add a point to the image. The selected one is highlighted in red
            Press:
                f: select next point
                z: remove selected point
                a, w, d, x: move selected point
                u: update the positions of the points
                
                o: zoom on image
                c: go back to previous view
                v: go forward to next view
                p: move the field of view in a zoomed view
                
                q: stop the interaction with the figure
        '''
        self.real_distance = real_distance
        
        fig, ax = plt.subplots()
        ax.imshow(img, **kwargs)
        
        def scatter_points():
            p = np.array(self.c_points)
            return ax.scatter(p[:,0],p[:,1],
                              marker='+',color='yellow')
        def highlight_point():
            self._c_constant = None
            return ax.scatter(self.c_points[self.c_current_index][0],self.c_points[self.c_current_index][1],
                              marker='+',color='red')
        
        def remove(obj):
            if obj:
                if obj.axes:
                    obj.remove()
        
        def onclick(event):
            
            if len(self.c_points) < 2:
                # works with right click
                if event.button == 3:
                    ix = int(event.xdata)
                    iy = int(event.ydata)

                    self.c_points.append([ix,iy])

                    #clear previous plots
                    remove(self.c_points_on_ax)
                    remove(self.c_highlight_on_ax)
                    self.c_points_on_ax = scatter_points()
                    self.c_current_index = len(self.c_points) - 1
                    self.c_highlight_on_ax = highlight_point()
            
        def onpress(event):
            
            # scroll points
            if event.key == 'f':
                self.c_current_index = (self.c_current_index + 1) % len(self.c_points)
                remove(self.c_highlight_on_ax)
                self.c_highlight_on_ax = highlight_point()
            
            # remove a point
            if event.key == 'z':
                remove(self.c_points_on_ax)
                remove(self.c_highlight_on_ax)
                _ = self.c_points.pop(self.c_current_index)
                self.c_points_on_ax = scatter_points()
                
            # move a point
                # left
            if event.key == 'a':
                remove(self.c_highlight_on_ax)
                self.c_points[self.c_current_index][0] -= 1
                self.c_highlight_on_ax = highlight_point()
                
                # right
            if event.key == 'd':
                remove(self.c_highlight_on_ax)
                self.c_points[self.c_current_index][0] += 1
                self.c_highlight_on_ax = highlight_point()
            
                # up
            if event.key == 'w':
                remove(self.c_highlight_on_ax)
                self.c_points[self.c_current_index][1] -= 1
                self.c_highlight_on_ax = highlight_point()
            
                # down
            if event.key == 'x':
                remove(self.c_highlight_on_ax)
                self.c_points[self.c_current_index][1] += 1
                self.c_highlight_on_ax = highlight_point()
                
            # update the points positions
            if event.key == 'u':
                remove(self.c_points_on_ax)
                self.c_points_on_ax = scatter_points()               
                
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        cid2 = fig.canvas.mpl_connect('key_press_event', onpress)
        
        return fig
        
    def measure_distance(self,img, **kwargs):
        '''
        Allows to measure the distance in mm between two points
        
        Input:
            img: image on the same scale as the one used for calibration
            
        Requirements:
            You need to have %matplotlib notebook
            The object must have been previously calibrated
            
        Controls:
            Right click to add a point to the image. The selected one is highlighted in red
            Press:
                f: select next point
                z: remove selected point
                a, w, d, x: move selected point
                u: update the positions of the points
                
                o: zoom on image
                c: go back to previous view
                v: go forward to next view
                p: move the field of view in a zoomed view
                
                q: stop the interaction with the figure
        '''
        try:
            _ = self.c_constant
        except IndexError:
            raise ValueError('Object must be calibrated to use this method')
        
        fig, ax = plt.subplots()
        ax.imshow(img, **kwargs)
        
        def scatter_points():
            p = np.array(self.d_points)
            if len(self.d_points) == 2:
                pixel_distance = np.sqrt((self.d_points[0][0] - self.d_points[1][0])**2 +
                                          (self.d_points[0][1] - self.d_points[1][1])**2)
                ax.set_title(f'distance = {self.c_constant*pixel_distance : .4f} mm')
            return ax.scatter(p[:,0],p[:,1],
                              marker='+',color='yellow')
        def highlight_point():
            return ax.scatter(self.d_points[self.d_current_index][0],self.d_points[self.d_current_index][1],
                              marker='+',color='red')
        
        def remove(obj):
            ax.set_title('')
            if obj:
                if obj.axes:
                    obj.remove()
        
        def onclick(event):
            
            if len(self.d_points) < 2:
                # works with right click
                if event.button == 3:
                    ix = int(event.xdata)
                    iy = int(event.ydata)

                    self.d_points.append([ix,iy])

                    #clear previous plots
                    remove(self.d_points_on_ax)
                    remove(self.d_highlight_on_ax)
                    self.d_points_on_ax = scatter_points()
                    self.d_current_index = len(self.d_points) - 1
                    self.d_highlight_on_ax = highlight_point()
            
        def onpress(event):
            
            # scroll points
            if event.key == 'f':
                self.d_current_index = (self.d_current_index + 1) % len(self.d_points)
                remove(self.d_highlight_on_ax)
                self.d_highlight_on_ax = highlight_point()
            
            # remove a point
            if event.key == 'z':
                remove(self.d_points_on_ax)
                remove(self.d_highlight_on_ax)
                _ = self.d_points.pop(self.d_current_index)
                self.d_points_on_ax = scatter_points()
                
            # move a point
                # left
            if event.key == 'a':
                remove(self.d_highlight_on_ax)
                self.d_points[self.d_current_index][0] -= 1
                self.d_highlight_on_ax = highlight_point()
                
                # right
            if event.key == 'd':
                remove(self.d_highlight_on_ax)
                self.d_points[self.d_current_index][0] += 1
                self.d_highlight_on_ax = highlight_point()
            
                # up
            if event.key == 'w':
                remove(self.d_highlight_on_ax)
                self.d_points[self.d_current_index][1] -= 1
                self.d_highlight_on_ax = highlight_point()
            
                # down
            if event.key == 'x':
                remove(self.d_highlight_on_ax)
                self.d_points[self.d_current_index][1] += 1
                self.d_highlight_on_ax = highlight_point()
                
            # update the points positions
            if event.key == 'u':
                remove(self.d_points_on_ax)
                self.d_points_on_ax = scatter_points()               
                
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        cid2 = fig.canvas.mpl_connect('key_press_event', onpress)
        
        return fig
    
    def find_channel(self,img, **kwargs):
        '''
        Allows to isolate the channel from the rest of the image
        
        Input:
            img: image with the channel
        
        Requirements:
            You need to have %matplotlib notebook
        
        Controls:
            Right click to add a point to the image. The selected one is highlighted in red
            Press:
                f: select next point
                z: remove selected point
                a, w, d, x: move selected point
                u: update the positions of the points
                
                b: to toggle the view of the channel the object will use for the analysis
                t: make the channel thicker if it is shown
                y: make the channel thinner if it is shown
                
                o: zoom on image
                c: go back to previous view
                v: go forward to next view
                p: move the field of view in a zoomed view
                
                q: stop the interaction with the figure
        '''
        fig, ax = plt.subplots()
        ax.imshow(img, **kwargs)
        
        def scatter_points():
            p = np.array(self.points)
            return ax.scatter(p[:,0],p[:,1],
                              marker='+',color='yellow')
        def highlight_point():
            self._central_ys = None
            self._s = None
            return ax.scatter(self.points[self.current_index][0],self.points[self.current_index][1],
                              marker='+',color='red')
        
        def remove(obj):
            if obj:
                if obj.axes:
                    obj.remove()
        
        def onclick(event):
            
            # works with right click
            if event.button == 3:
                ix = int(event.xdata)
                iy = int(event.ydata)

                self.points.append([ix,iy])

                #clear previous plots
                remove(self.points_on_ax)
                remove(self.highlight_on_ax)
                self.points_on_ax = scatter_points()
                self.current_index = len(self.points) - 1
                self.highlight_on_ax = highlight_point()
                self.sorted_points = False
            
        def onpress(event):
            
            # scroll points
            if event.key == 'f':
                self.current_index = (self.current_index + 1) % len(self.points)
                remove(self.highlight_on_ax)
                self.highlight_on_ax = highlight_point()
            
            # remove a point
            if event.key == 'z':
                remove(self.points_on_ax)
                remove(self.highlight_on_ax)
                _ = self.points.pop(self.current_index)
                self.points_on_ax = scatter_points()
                
            # move a point
                # left
            if event.key == 'a':
                remove(self.highlight_on_ax)
                self.points[self.current_index][0] -= 1
                self.highlight_on_ax = highlight_point()
                
                # right
            if event.key == 'd':
                remove(self.highlight_on_ax)
                self.points[self.current_index][0] += 1
                self.highlight_on_ax = highlight_point()
            
                # up
            if event.key == 'w':
                remove(self.highlight_on_ax)
                self.points[self.current_index][1] -= 1
                self.highlight_on_ax = highlight_point()
            
                # down
            if event.key == 'x':
                remove(self.highlight_on_ax)
                self.points[self.current_index][1] += 1
                self.highlight_on_ax = highlight_point()
                
            # update the points positions
            if event.key == 'u':
                remove(self.points_on_ax)
                self.points_on_ax = scatter_points()
                
            # show boundaries
            if event.key == 'b':
                if self.shown_borders:
                    remove(self.u_border_on_ax)
                    remove(self.l_border_on_ax)
                    self.shown_borders = not self.shown_borders
                    
                else:
                    # sort the points
                    if not self.sorted_points:
                        dtype = [('x',int),('idx',int)]
                        a = np.array([(x,i) for i,x in enumerate(np.array(self.points)[:,0])], dtype=dtype)
                        a = np.sort(a,order='x')
                        self.points = [self.points[i] for i in a['idx']]
                        self.sorted_points = True
                        
                        
                    remove(self.u_border_on_ax)
                    remove(self.l_border_on_ax)
                    xs = np.array(self.points)[:,0]
                    upper_ys = np.array(self.points)[:,1] - self.thickness
                    lower_ys = upper_ys + 2*self.thickness

                    
                    self.u_border_on_ax, = ax.plot(xs,upper_ys,color='yellow')
                    self.l_border_on_ax, = ax.plot(xs,lower_ys,color='yellow')
                    
                    self.shown_borders = not self.shown_borders
                    
            # increase thickness of the borders
            if event.key == 't':
                if self.shown_borders:                    
                    self.thickness += 1
                    
                    xs = np.array(self.points)[:,0]
                    upper_ys = np.array(self.points)[:,1] - self.thickness
                    lower_ys = upper_ys + 2*self.thickness

                    remove(self.l_border_on_ax)
                    remove(self.u_border_on_ax)
                    self.u_border_on_ax, = ax.plot(xs,upper_ys,color='yellow')
                    self.l_border_on_ax, = ax.plot(xs,lower_ys,color='yellow')
                    
            # decrease thickness of the borders
            if event.key == 'y':
                if self.shown_borders and self.thickness > 1:                    
                    self.thickness -= 1
                    
                    xs = np.array(self.points)[:,0]
                    upper_ys = np.array(self.points)[:,1] - self.thickness
                    lower_ys = upper_ys + 2*self.thickness

                    remove(self.l_border_on_ax)
                    remove(self.u_border_on_ax)
                    self.u_border_on_ax, = ax.plot(xs,upper_ys,color='yellow')
                    self.l_border_on_ax, = ax.plot(xs,lower_ys,color='yellow')
                     
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        cid2 = fig.canvas.mpl_connect('key_press_event', onpress)
        
        return fig