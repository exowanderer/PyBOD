
# coding: utf-8

# # Bayesian Object Detection with PyMultinest

# In[1]:

from __future__ import absolute_import, unicode_literals, print_function
import pymultinest
import math
import os
import threading, subprocess
from sys import platform

from pylab import *;ion()

if not os.path.exists("chains"): os.mkdir("chains")


# ** MultiModel Gaussian Normal in 1D**

# In[2]:

get_ipython().magic('matplotlib inline')
from pylab import *;ion()

from pymultinest.solve import Solver,solve
from numpy import pi, sin, cos, linspace

def gaussian1Dp(cube):
    center = cube[0]
    width  = cube[1]
    height = cube[2]
    return lambda y: height*np.exp(-0.5*(( (center - y) / width)**2))# / sqrt(2*pi*width**2)

def gaussian1D(cube):
    center = cube[0]
    width  = cube[1]
    return lambda y: np.exp(-0.5*(( (center - y) / width)**2)) / sqrt(2*pi*width**2)

def straight_line(cube):
    offset = cube[0]
    slope  = cube[1]
    return lambda abscissa: offset + slope * abscissa

def sine_wave(cube):
    amp    = cube[0]
    period = cube[1]
    return lambda abscissa: amp*sin(2*pi / period * abscissa)

np.random.seed(42)

param0a= -0.5#0.05
param0b= 0.5#0.05
param1a= 0.1#5*pi
param1b= 0.1#5*pi
param2a= 0.8
param2b= 0.8

yunc  = 0.1
nPts  = int(100)
nThPts= int(1e3)

xmin  = -1#*pi
xmax  =  1#*pi
dx    = 0.1*(xmax - xmin)

# model = sine_wave; parameters = ["amp", "period"]
# model = straight_line; parameters = ["offset", "slope"]
# model = gaussian1D; parameters = ["center", "width"]
model = gaussian1Dp; parameters = ["center", "width", "height"]

yuncs = np.random.normal(yunc, 1e-2 * yunc, nPts)
thdata= np.linspace(xmin-dx, xmax+dx, nThPts)

xdata = np.linspace(xmin,xmax,nPts)
# xdata = np.random.uniform(xmin, xmax, nPts)
# xdata = sort(xdata)

ydata = model([param0a,param1a,param2a])(xdata) + model([param0b,param1b,param2b])(xdata)

yerr  = np.random.normal(0, yuncs, nPts)
zdata = ydata + yerr

figure(figsize=(10,10))
plot(thdata, model([param0a,param1a,param2a])(thdata) + model([param0b,param1b,param2b])(thdata))
errorbar(xdata, zdata, yunc*ones(zdata.size), fmt='o')


# In[3]:

def prior(cube, ndim, nparams):
    cube[0] = cube[0]*2 - 1
    cube[1] = cube[1]*2
    cube[2] = cube[2]*2
    pass

def loglike(cube, ndim, nparams):
    modelNow = model(cube)(xdata)
    return -0.5*((modelNow - ydata)**2. / yuncs**2.).sum()


# In[4]:

if not os.path.exists("chains"): os.mkdir("chains")

n_params = len(parameters)

plt.figure(figsize=(5*n_params, 5*n_params))

# we want to see some output while it is running
progress = pymultinest.ProgressPlotter(n_params = n_params, outputfiles_basename='chains/2-'); progress.start()
# threading.Timer(2, show, ["chains/2-phys_live.points.pdf"]).start() # delayed opening
# run MultiNest
pymultinest.run(loglike, prior, n_params, importance_nested_sampling = False, resume = False, verbose = True,             sampling_efficiency = 'model', n_live_points = 1000, outputfiles_basename='chains/2-')

# ok, done. Stop our progress watcher
progress.stop()

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='chains/2-')
s = a.get_stats()


# In[5]:

import json

# store name of parameters, always useful
with open('%sparams.json' % a.outputfiles_basename, 'w') as f:
    json.dump(parameters, f, indent=2)
# store derived stats
with open('%sstats.json' % a.outputfiles_basename, mode='w') as f:
    json.dump(s, f, indent=2)

print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))


# In[6]:

import matplotlib.pyplot as plt
plt.clf()

# Here we will plot all the marginals and whatnot, just to show off
# You may configure the format of the output here, or in matplotlibrc
# All pymultinest does is filling in the data of the plot.

# Copy and edit this file, and play with it.

p = pymultinest.PlotMarginalModes(a)
plt.figure(figsize=(5*n_params, 5*n_params))
#plt.subplots_adjust(wspace=0, hspace=0)
for i in range(n_params):
    plt.subplot(n_params, n_params, n_params * i + i + 1)
    p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
    plt.ylabel("Probability")
    plt.xlabel(parameters[i])
    
    for j in range(i):
        plt.subplot(n_params, n_params, n_params * j + i + 1)
        #plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
        p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
        plt.xlabel(parameters[i])
        plt.ylabel(parameters[j])

# plt.savefig("chains/marginals_multinest.pdf") #, bbox_inches='tight')
# show("chains/marginals_multinest.pdf")

plt.figure(figsize=(5*n_params, 5*n_params))
plt.subplot2grid((5*n_params, 5*n_params), loc=(0,0))
for i in range(n_params):
    #plt.subplot(n_params, n_params, i + 1)
    # outfile = '%s-mode-marginal-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_modes_marginal(i, with_ellipses = True, with_points = False)
    plt.ylabel("Probability")
    plt.xlabel(parameters[i])
    # plt.savefig(outfile, format='pdf', bbox_inches='tight')
    # plt.close()
    
    # outfile = '%s-mode-marginal-cumulative-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
    plt.ylabel("Cumulative probability")
    plt.xlabel(parameters[i])
    # plt.savefig(outfile, format='pdf', bbox_inches='tight')
    # plt.close()

print("Take a look at the pdf files in chains/") 


# In[7]:

print('best\t', np.round(p.analyser.get_best_fit()['parameters'],3))
for k,mode in enumerate(p.analyser.get_stats()['modes']):
    print('mode' + str(k) + '\t', np.round(mode['mean'],3))

print('True a\t', [param0a, param1a])
print('True b\t', [param0b, param1b])


# In[8]:

p.analyser.get_stats()


# In[9]:

figure(figsize=(10,10))
errorbar(xdata, zdata, yunc*ones(zdata.size), fmt='o')
modelAll = np.zeros(thdata.size)
for m in p.analyser.get_stats()['modes']:
    modelAll = modelAll + model(m['mean'])(thdata)
    plot(thdata, model(m['mean'])(thdata))

plot(thdata, modelAll)
plot(thdata, model(p.analyser.get_best_fit()['parameters'])(thdata))


# ** MultiModel Gaussian Normal in 2D**

# In[10]:

get_ipython().magic('matplotlib inline')
from pylab import *;ion()

from pymultinest.solve import Solver,solve
from numpy import pi, sin, cos, linspace

def gaussian2D(cube):
    center = cube[0]
    width  = cube[1]
    return lambda y,x: np.exp(-0.5*((( (center - y) / width)**2) + (( (center - x) / width)**2))) / sqrt(2*pi*width**2)

np.random.seed(42)

param0a= 0.75#0.05
param1a= 0.05#0.05
param0b= 0.25#0.05
param1b= 0.05#0.05

# param2= 0.8

yunc  = 0.1
nPts  = int(100)
nThPts= int(1e3)

xmin  = -0#*pi
xmax  =  1#*pi
dx    = 0.1*(xmax - xmin)

ymin  = -0#*pi
ymax  =  1#*pi
dy    = 0.1*(ymax - ymin)

model = gaussian2D; parameters = ["center", "width"]

yuncs = np.random.normal(yunc, 1e-2 * yunc, (nPts,nPts))
# thdata= np.linspace(xmin-dx, xmax+dx, nThPts)

xdata = np.ones((nPts,nPts))*np.linspace(xmin,xmax,nPts)
ydata = (np.ones((nPts,nPts))*np.linspace(ymin,ymax,nPts)).T

zmodel  = model([param0a,param1a])(ydata,xdata) + model([param0b,param1b])(ydata,xdata)
zerr    = np.random.normal(0, yuncs, (nPts,nPts))
zdata   = zmodel + zerr

figure(figsize=(10,10))
imshow(zdata, extent=[xdata.min(), xdata.max(), ydata.min(), ydata.max()])


# In[11]:

# our probability functions
# Taken from the eggbox problem.
# model = sine_wave; parameters = ["amp", "period"]
# model = gaussian1D; parameters = ["center", "width"]
# model = straight_line; parameters = ["offset", "slope"]

def prior(cube, ndim, nparams):
    pass

def loglike(cube, ndim, nparams):
    modelNow = gaussian2D(cube)(ydata,xdata)
    return -0.5*((modelNow - zdata)**2. / yuncs**2.).sum()


# In[12]:

if not os.path.exists("chains"): os.mkdir("chains")

# number of dimensions our problem has
# parameters = ["x", "y"]
n_params = len(parameters)

plt.figure(figsize=(5*n_params, 5*n_params))
# we want to see some output while it is running
progress = pymultinest.ProgressPlotter(n_params = n_params, outputfiles_basename='chains/2-'); progress.start()
# threading.Timer(2, show, ["chains/2-phys_live.points.pdf"]).start() # delayed opening
# run MultiNest
pymultinest.run(loglike, prior, n_params, importance_nested_sampling = False,                 resume = False, verbose = True, sampling_efficiency = 'model', n_live_points = 1000,                 outputfiles_basename='chains/2-')

# ok, done. Stop our progress watcher
progress.stop()

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='chains/2-')
s = a.get_stats()

# fig = gcf()
# axs  = fig.get_axes()
# for ax in axs:
#     ax.set_ylim(-16,0)


# In[13]:

import json

# store name of parameters, always useful
with open('%sparams.json' % a.outputfiles_basename, 'w') as f:
    json.dump(parameters, f, indent=2)
# store derived stats
with open('%sstats.json' % a.outputfiles_basename, mode='w') as f:
    json.dump(s, f, indent=2)

print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))


# In[14]:

import matplotlib.pyplot as plt
plt.clf()

# Here we will plot all the marginals and whatnot, just to show off
# You may configure the format of the output here, or in matplotlibrc
# All pymultinest does is filling in the data of the plot.

# Copy and edit this file, and play with it.

p = pymultinest.PlotMarginalModes(a)
plt.figure(figsize=(5*n_params, 5*n_params))
#plt.subplots_adjust(wspace=0, hspace=0)
for i in range(n_params):
    plt.subplot(n_params, n_params, n_params * i + i + 1)
    p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
    plt.ylabel("Probability")
    plt.xlabel(parameters[i])
    
    for j in range(i):
        plt.subplot(n_params, n_params, n_params * j + i + 1)
        #plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
        p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
        plt.xlabel(parameters[i])
        plt.ylabel(parameters[j])

# plt.savefig("chains/marginals_multinest.pdf") #, bbox_inches='tight')
# show("chains/marginals_multinest.pdf")

plt.figure(figsize=(5*n_params, 5*n_params))
plt.subplot2grid((5*n_params, 5*n_params), loc=(0,0))
for i in range(n_params):
    #plt.subplot(n_params, n_params, i + 1)
    # outfile = '%s-mode-marginal-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_modes_marginal(i, with_ellipses = True, with_points = False)
    plt.ylabel("Probability")
    plt.xlabel(parameters[i])
    # plt.savefig(outfile, format='pdf', bbox_inches='tight')
    # plt.close()
    
    # outfile = '%s-mode-marginal-cumulative-%d.pdf' % (a.outputfiles_basename,i)
    p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
    plt.ylabel("Cumulative probability")
    plt.xlabel(parameters[i])
    # plt.savefig(outfile, format='pdf', bbox_inches='tight')
    # plt.close()

print("Take a look at the pdf files in chains/") 


# In[15]:

print('best\t', np.round(p.analyser.get_best_fit()['parameters'],3))
for k,mode in enumerate(p.analyser.get_stats()['modes']):
    print('mode' + str(k) + '\t', np.round(mode['mean'],3),'\t', np.round(mode['local log-evidence'],3))

print('True a\t', [param0a, param1a])
print('True b\t', [param0b, param1b])


# In[16]:

modelAll = np.zeros((nPts, nPts))
fig=figure(figsize=(20,10))
for km,mode in enumerate(p.analyser.get_stats()['modes']):
    modelAll = modelAll + model(mode['mean'])(ydata,xdata)
    ax = fig.add_subplot(1,len(p.analyser.get_stats()['modes']), km+1)
    ims = ax.imshow(model(mode['mean'])(ydata,xdata))
    plt.colorbar(ims)

fig = figure(figsize=(20,10))
ax = fig.add_subplot(131)
ims = ax.imshow(modelAll)
plt.colorbar(ims)
ax = fig.add_subplot(132)
ims = ax.imshow(model(p.analyser.get_best_fit()['parameters'])(ydata,xdata))
plt.colorbar(ims)
ax = fig.add_subplot(133)
ims = ax.imshow(zdata)
plt.colorbar(ims)

# Residuals
modelAll = np.zeros((nPts, nPts))
for km,mode in enumerate(p.analyser.get_stats()['modes']):
    if np.round(mode['local log-evidence'],3) > -400000.:
        modelAll = modelAll + model(mode['mean'])(ydata,xdata)

fig = figure(figsize=(20,10))
ax  = fig.add_subplot(131)
ims = ax.imshow(zdata - modelAll)
plt.colorbar(ims)
ax  = fig.add_subplot(132)
ax.hist((zdata - modelAll).ravel(), bins=1000, normed=True);
# plt.colorbar(ims)


# In[17]:

p.analyser.get_stats()


# In[ ]:



