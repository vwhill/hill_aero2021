#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:44:58 2020

@author: vincenthill

Function to generate and discretize linearized lateral aircraft model
"""
#%% Modules

import astar_single as astar
import guideclass as guide
import numpy as np
import matplotlib.pyplot as plt

#%% Build dynamic model

dt = 0.01
mod = 100
maxtime = 5000
maxiter = np.int(maxtime/dt)
t = np.linspace(0, maxtime, maxiter)

fw = guide.LateralFixedWing(dt)
F = fw.F
G = fw.G
statedim = fw.statedim
indim = fw.indim

sys = guide.LQI(dt, F, G, statedim, indim)

Fi = sys.Fi
Gi = sys.Gi

#%% Define targets and mission area

xa = 0
ya = 0
xa_i = xa
ya_i = ya

a = 100
b = 100
mapsize = 3000
mesh = astar.meshgen(a, b, mapsize)

targ = np.array([[mesh[50, 0], mesh[0, 0]],
                 [mesh[0, 0], mesh[0, 50]],
                 [mesh[50, 0], mesh[0, 50]],
                 [mesh[0, 0], mesh[0, 0]],
                 [mesh[0, 0], mesh[0, 50]],
                 [mesh[25, 0], mesh[75, 0]],
                 [mesh[50, 0], mesh[0, 50]],
                 [mesh[50, 0], mesh[0, 0]]])

# targ = np.array([[1000, 1000],
#                  [0, 2000],
#                  [-1000, 1000],
#                  [0, 0]])

# targ = 3000*np.random.rand(10, 2)

# targ = np.array([np.linspace(1,1000,num=10),np.linspace(1,500,num=10)]).T

xt = targ[0, 0]
yt = targ[0, 1]
wpt = 0

xy = np.zeros((2, maxiter))
xy[0, 0] = xa
xy[1, 0] = ya

[d2d_i, Hdes_i] = guide.guidance(xa, ya, xt, yt)

dxy = np.zeros((1, maxiter))
dxy[0, 0] = d2d_i

#%% Simulate

x = np.zeros([statedim*2, 1])
x[0] = 3.0
x[4] = np.deg2rad(0)
v = x[0, 0]
psi = x[4, 0]
u = 30.0

xdes = np.zeros([statedim*2, 1])
xdes[4, 0] = Hdes_i
xs = np.zeros((statedim*2, maxiter))
xs[:, 0] = np.ndarray.flatten(x)

for ii in range(0, maxiter):
    if ii % mod == 0:
        [d2d, Hdes] = guide.guidance(xa, ya, xt, yt)
        dxy[0, ii] = d2d
        xdes[4] = Hdes
        x = Fi@x+Gi@xdes
        xs[:, ii] = np.ndarray.flatten(x)
        v = x[0, 0]
        psi = x[4, 0]
        [xa, ya] = guide.velprop(xa, ya, u, v, psi, dt)
        xy[0, ii] = xa
        xy[1, ii] = ya
        if d2d < 50:
            wpt = wpt+1
            if wpt == np.size(targ, axis=0):
                np.disp('Im Finished!')
                break
            else:
                xt = targ[wpt, 0]
                yt = targ[wpt, 1]
                [d2d, Hdes] = guide.guidance(xa, ya, xt, yt)
                dxy[0, ii] = d2d
                xdes[4] = Hdes
    else:
        x = Fi@x+Gi@xdes
        xs[:, ii] = np.ndarray.flatten(x)
        v = x[0, 0]
        [xa, ya] = guide.velprop(xa, ya, u, v, psi, dt)
        xy[0, ii] = xa
        xy[1, ii] = ya
        
#%% Plots

plt.plot(xy[0, :ii], xy[1, :ii])
plt.plot(xa_i, ya_i, 'ro')
plt.plot(targ[0, 0], targ[0, 1], 'rx')
plt.plot(targ[1, 0], targ[1, 1], 'rx')
plt.plot(targ[2, 0], targ[2, 1], 'rx')
plt.plot(targ[3, 0], targ[3, 1], 'rx')
plt.plot(targ[4, 0], targ[4, 1], 'rx')
plt.plot(targ[5, 0], targ[5, 1], 'rx')
plt.plot(targ[6, 0], targ[6, 1], 'rx')
plt.plot(targ[7, 0], targ[7, 1], 'rx')
plt.show()

# plt.plot(t, np.ndarray.flatten(dxy))
# plt.show()