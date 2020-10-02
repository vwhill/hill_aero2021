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
maxtime = 10000
maxiter = np.int(maxtime/dt)
t = np.linspace(0, maxtime, maxiter)

fw = guide.LateralFixedWing(dt)
statedim = fw.statedim
indim = fw.indim

sysi = guide.LQI(dt, fw.F, fw.G, statedim, indim)
Fi = sysi.Fi
Gi = sysi.Gi

sysr = guide.LQR(dt, fw.F, fw.G, statedim, indim)
Fr = sysr.Fr
Gr = sysr.Gr

#%% Define targets and mission area

a = 50
b = 50
mapsize = 10000
[mesh, xm, ym] = astar.meshgen(a, b, mapsize)

start = np.array([[0, 0]])
xa = start
ya = start
xa_i = xa
ya_i = ya

end = np.array([[25, 25]])

[maze, cost, obs] = astar.mazegen(a, b)

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

xt = targ[0, 0]
yt = targ[0, 1]
wpt = 0

xy = np.zeros((2, maxiter))
xy[0, 0] = start
xy[1, 0] = start

[d2d_i, Hdes_i] = guide.guidance(xa, ya, xt, yt)

dxy = np.zeros((1, maxiter))
dxy[0, 0] = d2d_i

#%% Simulate LQR

x = np.zeros([statedim, 1])
x[0] = 3.0
x[4] = np.deg2rad(Hdes_i)
v = x[0, 0]
psi = x[4, 0]
u = 10.0

xdes = np.zeros([statedim, 1])
xdes[4, 0] = Hdes_i
xs = np.zeros((statedim, maxiter))
xs[:, 0] = np.ndarray.flatten(x)

for ii in range(0, maxiter):
    if ii % mod == 0:
        [d2d, Hdes] = guide.guidance(xa, ya, xt, yt)
        dxy[0, ii] = d2d
        xdes[4] = Hdes
        x = Fr@x+Gr@xdes
        xs[:, ii] = np.ndarray.flatten(x)
        v = x[0, 0]
        psi = x[4, 0]
        [xa, ya] = guide.velprop(xa, ya, u, v, psi, dt)
        xy[0, ii] = xa
        xy[1, ii] = ya
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
    else:
        x = Fr@x+Gr@xdes
        xs[:, ii] = np.ndarray.flatten(x)
        v = x[0, 0]
        [xa, ya] = guide.velprop(xa, ya, u, v, psi, dt)
        xy[0, ii] = xa
        xy[1, ii] = ya

#%% Plots

plt.plot(xy[0, :ii], xy[1, :ii], label='Aircraft path', linewidth=3)
plt.scatter(targ[:, 0], targ[:, 1], s=5, label='Waypoints', color='magenta')
plt.plot(xa_i, ya_i, 'ro')
plt.plot(xm[end[1]], ym[end[0]], 'rx')
plt.scatter(xm[obsx], ym[obsy], s=30, label='Obstacles', color='black')
plt.xlim(-0.05*mapsize, mapsize+mapsize*0.05)
plt.ylim(-0.05*mapsize, mapsize+mapsize*0.05)
plt.legend()
plt.show()

# plt.plot(t, np.ndarray.flatten(dxy))
# plt.show()

# %% Dead code

# targ = np.array([[mesh[50, 0], mesh[0, 0]],
#                  [mesh[0, 0], mesh[0, 50]],
#                  [mesh[50, 0], mesh[0, 50]],
#                  [mesh[0, 0], mesh[0, 0]],
#                  [mesh[0, 0], mesh[0, 50]],
#                  [mesh[25, 0], mesh[75, 0]],
#                  [mesh[50, 0], mesh[0, 50]],
#                  [mesh[50, 0], mesh[0, 0]]])

# targ = np.array([[1000, 1000],
#                  [0, 2000],
#                  [-1000, 1000],
#                  [0, 0]])

# targ = 3000*np.random.rand(10, 2)

# targ = np.array([[0, 1000]])


# targ = np.array([np.linspace(1,1000,num=10),np.linspace(1,500,num=10)]).T

# targ = np.array([np.linspace(1,1000,num=10),np.linspace(1,500,num=10)]).T

# x = np.zeros([statedim*2, 1])
# x[0] = 3.0
# x[4] = np.deg2rad(0)
# v = x[0, 0]
# psi = x[4, 0]
# u = 30.0

# xdes = np.zeros([statedim*2, 1])
# xdes[4, 0] = Hdes_i
# xs = np.zeros((statedim*2, maxiter))
# xs[:, 0] = np.ndarray.flatten(x)

# for ii in range(0, maxiter):
#     if ii % mod == 0:
#         [d2d, Hdes] = guide.guidance(xa, ya, xt, yt)
#         dxy[0, ii] = d2d
#         xdes[4] = Hdes
#         x = Fi@x+Gi@xdes
#         if np.abs(x[9]) > 1.99*np.pi:
#             x[9] = 0
#         xs[:, ii] = np.ndarray.flatten(x)
#         v = x[0, 0]
#         psi = x[4, 0]
#         [xa, ya] = guide.velprop(xa, ya, u, v, psi, dt)
#         xy[0, ii] = xa
#         xy[1, ii] = ya
#         if d2d < 50:
#             wpt = wpt+1
#             x[5:] = np.zeros((np.size(x[5:]), 1)) # reset error buildup
#             if wpt == np.size(targ, axis=0):
#                 np.disp('Im Finished!')
#                 break
#             else:
#                 xt = targ[wpt, 0]
#                 yt = targ[wpt, 1]
#                 [d2d, Hdes] = guide.guidance(xa, ya, xt, yt)
#                 dxy[0, ii] = d2d
#                 xdes[4] = Hdes
#     else:
#         x = Fi@x+Gi@xdes
#         if np.abs(x[9]) > 1.99*np.pi: # reset error buildup
#             x[9] = 0
#         xs[:, ii] = np.ndarray.flatten(x)
#         v = x[0, 0]
#         [xa, ya] = guide.velprop(xa, ya, u, v, psi, dt)
#         xy[0, ii] = xa
#         xy[1, ii] = ya
# pathidx = np.where(path>-1)
# pathidx_x = astar.selection_sort(np.asarray(pathidx[1]))
# pathidx_y = astar.selection_sort(np.asarray(pathidx[0]))
# targ = np.zeros((np.size(pathidx[0]), 2))
# 
# for jj in range(0, np.size(pathidx_x)):
#     targ[jj, 0] = xm[pathidx_x[jj]]
#     targ[jj, 1] = ym[pathidx_y[jj]]