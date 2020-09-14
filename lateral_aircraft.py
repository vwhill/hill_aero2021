#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:44:58 2020

@author: vincenthill

Function to generate and discretize linearized lateral aircraft model
"""
#%% Modules

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal as sig
from scipy import linalg as lin

#%%  Classes

def discretize(dt, A, B, C, D):
    sys = sig.StateSpace(A, B, C, D)
    sys = sys.to_discrete(dt)
    F = sys.A
    G = sys.B
    return F, G
    
class LateralFixedWing:
    def __init__(self, dt):
        self.statedim = 5
        self.indim = 2
                        #     v       p       r     phi    psi
        self.A = np.array([[-2.382,   0.,   -30.1,  65.49, 0.],  # v
                           [-0.702, -16.06,  0.872, 0.,    0.],  # p
                           [ 0.817, -16.65, -3.54,  0.,    0.],  # r
                           [ 0.,      1.,    0.,    0.,    0.],  # phi
                           [ 0.,      0.,    1.,    0.,    0.]]) # psi
                
                           # ail      rud 
        self.B = np.array([[ 0.,    -7.41],  # v
                           [-36.3,  -688.],  # p
                           [-0.673, -68.0],  # r
                           [ 0.,      0.],   # phi
                           [ 0.,      0.]])  # psi

        self.C = np.eye(self.statedim)
        self.D = np.eye(self.statedim, self.indim)
        [self.F, self.G] = discretize(dt, self.A, self.B, self.C, self.D)

class LQI:
    def __init__(self, dt, F, G, statedim, indim):
        Fe1 = np.concatenate((F, np.zeros((statedim, statedim))), axis=1)
        Fe2 = np.concatenate((-dt*np.eye(statedim), np.eye(statedim)), axis=1)
        Fe = np.vstack((Fe1, Fe2))
        
        Ge = np.vstack((G, np.zeros((statedim, indim))))
        
        Ce = np.identity(statedim*2)
        De = np.zeros((statedim*2, statedim*2))
        
        Q = np.identity(statedim*2)
        Q[0, 0] = 1.0               # v
        Q[1, 1] = 3.3               # p
        Q[2, 2] = (1.8/1.0)**2      # r
        Q[3, 3] = 1.0               # phi
        Q[4, 4] = 1.0               # psi
        Q[5, 5] = (1.0/1000.0)**2   # error states
        Q[6, 6] = (1.0/1000.0)**2
        Q[7, 7] = (1.0/980.0)**2
        Q[8, 8] = (1.0/1000.0)**2
        Q[9, 9] = (1.0/1000.0)**2
        
        R = 1.0*np.identity(indim)
        
        P = lin.solve_discrete_are(Fe, Ge, Q, R, e=None, s=None, balanced=True)
        K = lin.inv(Ge.T@P@Ge+R)@(Ge.T@P@Fe)
        
        sysac = sig.StateSpace(Fe-Ge@K, Ge@K, Ce, De, dt=dt)
        
        self.Fi = sysac.A
        self.Gi = sysac.B

def guidance(xa, ya, xt, yt):
    d2d = float(np.sqrt((xa-xt)**2+(ya-yt)**2))
    Hdes = float(-math.atan2((yt-ya), (xt-xa)))
    return d2d, Hdes

def velprop(xe, ye, u, v, psi, dt):
    uv = np.array(([float(u)], [float(v)]))
    dcm = np.array([[float(np.cos(psi)), float(np.sin(psi)),],
                    [float(-np.sin(psi)), float(np.cos(psi))]])
    xy = dcm@uv
    xp = float(xe+dt*xy[0])
    yp = float(ye+dt*xy[1])
    return xp, yp

#%% Simulation

dt = 0.01
mod = 100
maxtime = 1000
maxiter = np.int(maxtime/dt)
t = np.linspace(0, maxtime, maxiter)

fw = LateralFixedWing(dt)
F = fw.F
G = fw.G
statedim = fw.statedim
indim = fw.indim

sys = LQI(dt, F, G, statedim, indim)

Fi = sys.Fi
Gi = sys.Gi

xa = 0
ya = 0
xa_i = xa
ya_i = ya

# targ = np.array([[1000, 1000],
#                  [500, 2000],
#                  [3000, -1500]])

targ = 1000*np.random.rand(5, 2)

xt = targ[0, 0]
yt = targ[0, 1]
wpt = 0

xy = np.zeros((2, maxiter))
xy[0, 0] = xa
xy[1, 0] = ya

[d2d_i, Hdes_i] = guidance(xa, ya, xt, yt)

dxy = np.zeros((1, maxiter))
dxy[0, 0] = d2d_i

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
        [d2d, Hdes] = guidance(xa, ya, xt, yt)
        dxy[0, ii] = d2d
        xdes[4] = Hdes
        x = Fi@x+Gi@xdes
        xs[:, ii] = np.ndarray.flatten(x)
        v = x[0, 0]
        psi = x[4, 0]
        [xa, ya] = velprop(xa, ya, u, v, psi, dt)
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
                [d2d, Hdes] = guidance(xa, ya, xt, yt)
                dxy[0, ii] = d2d
                xdes[4] = Hdes
    else:
        x = Fi@x+Gi@xdes
        xs[:, ii] = np.ndarray.flatten(x)
        v = x[0, 0]
        [xa, ya] = velprop(xa, ya, u, v, psi, dt)
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
plt.show()

# plt.plot(t, np.ndarray.flatten(dxy))
# plt.show()