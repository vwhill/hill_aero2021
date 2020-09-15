# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:53:39 2020

@author: vince

Classes and functions to do 2D guidance
"""

import numpy as np
import math
from scipy import signal as sig
from scipy import linalg as lin

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