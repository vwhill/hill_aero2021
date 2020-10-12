# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:18:28 2020

@author: vince
"""

#%% Imports

import numpy as np
import gncpy.filters as filters
import gasur.swarm_estimator.tracker as trkr
import guideclass as guide
import astar_single as astar
import matplotlib.pyplot as plt

#%% Initialize

dt = 0.01
dyn = guide.LateralFixedWing(dt)

statedim = dyn.statedim
indim = dyn.indim

sysr = guide.LQR(dt, dyn.F, dyn.G, statedim, indim)
Fr = sysr.Fr
Gr = sysr.Gr
Hr = np.eye(statedim)
pnoise = 0.01*np.ones((statedim, 1))
mnoise = 0.01*np.ones((statedim, 1))
Q = np.eye(statedim)
R = np.eye(statedim)

for jj in range(0, statedim):
    Q[jj, jj] = pnoise[jj]
    R[jj, jj] = mnoise[jj]

kf = filters.KalmanFilter()

kf.set_proc_noise(mat=Q)
kf.meas_noise = R

kf.set_state_mat(mat=Fr)
kf.set_input_mat(mat=Gr)
kf.set_meas_mat(mat=Hr)

rfs = trkr.GeneralizedLabeledMultiBernoulli()
rfs.prob_detection = 0.95

#%% Define targets and mission area

maxtime = 5000
maxiter = np.int(maxtime/dt)
t = np.linspace(0, maxtime, maxiter)

a = 51
b = 51
mapsize = 10000
[mesh, xm, ym] = astar.meshgen(a, b, mapsize)

start = np.array([0, 0])
xa = start[1]
ya = start[0]
xa_i = xa
ya_i = ya

end = np.array([45, 45])

[maze, cost] = astar.mazegen(a, b)
maze[start[0], start[1]] = 0
maze[end[0], end[1]] = 0

obs = np.where(maze==1)
obsx = np.asarray(obs[1])
obsy = np.asarray(obs[0])

path_list = astar.search(maze, cost, start, end)
path = np.asarray(path_list)

idx = np.zeros((np.max(path), 2))
targ = np.zeros((np.max(path), 2))
pt = 0

for qq in range(0, np.max(path)):
    where = np.where(path==pt)
    idx[qq, 0] = where[1]
    idx[qq, 1] = where[0]
    pt = pt+1
    targ[qq, 0] = xm[np.int(idx[qq, 0])]
    targ[qq, 1] = ym[np.int(idx[qq, 1])]

xt = targ[1, 0]
yt = targ[1, 1]
wpt = 0

xy = np.zeros((2, maxiter))
xy[0, 0] = start[1]
xy[1, 0] = start[0]

[d2d_i, Hdes_i] = guide.guidance(xa, ya, xt, yt)

dxy = np.zeros((1, maxiter))
dxy[0, 0] = d2d_i

#%% Simulate LQR

x = np.zeros([statedim, 1])
x[0] = 1.0
x[4] = np.deg2rad(Hdes_i)
x0 = x+0.1*np.random.rand(statedim, 1)
cov = (x-x0)*(x-x0).T
kf.cov = cov
v = x[0, 0]
psi = x[4, 0]
u = 10.0

xdes = np.zeros([statedim, 1])
xdes[4, 0] = Hdes_i
xs = np.zeros((statedim, maxiter))
xs[:, 0] = np.ndarray.flatten(x)

for ii in range(0, maxiter):
    x = Fr@x+Gr@xdes+pnoise
    y = Hr@x+mnoise
    kf.cov = Fr@kf.cov@Fr.T+Q
    cor = kf.correct(cur_state=x, meas=y)
    x = cor[0]
    xs[:, ii] = np.ndarray.flatten(x)
    v = x[0, 0]
    psi = x[4, 0]
    [xa, ya] = guide.velprop(xa, ya, u, v, psi, dt)
    xy[0, ii] = xa
    xy[1, ii] = ya
    [d2d, Hdes] = guide.guidance(xa, ya, xt, yt)
    dxy[0, ii] = d2d
    xdes[4] = Hdes
    if d2d < 100:
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

#%% Plots

plt.plot(xy[0, :ii], xy[1, :ii], label='Aircraft path', linewidth=3)
plt.scatter(targ[:, 0], targ[:, 1], s=5, label='Waypoints', color='magenta')
plt.plot(xa_i, ya_i, 'ro')
plt.plot(xm[end[1]], ym[end[0]], 'rx')
plt.scatter(xm[obsx], ym[obsy], s=30, label='Obstacles', color='black')
plt.xlim(-0.05*mapsize, mapsize+mapsize*0.05)
plt.ylim(-0.05*mapsize, mapsize+mapsize*0.05)
# plt.legend()
plt.show()