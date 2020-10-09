# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:39:20 2020

@author: vince
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

sysr = guide.LQR(dt, fw.F, fw.G, statedim, indim)
Fr = sysr.Fr
Gr = sysr.Gr

#%% Initialize mission

a = 51
b = 51
mapsize = 10000
[mesh, xm, ym] = astar.meshgen(a, b, mapsize)

start = np.array([[0, 0],
                  [10, 0], 
                  [20, 0],
                  [30, 0],
                  [40, 0],
                  [50, 0]])

end = np.array([[25, 25],
                [5, 50],
                [15, 50],
                [25, 50],
                [35, 50],
                [45, 50]])

[maze, cost] = astar.mazegen(a, b)

for hh in range(np.size(start, axis=0)):
    maze[start[hh, 0]] = 0
    maze[start[hh, 1]] = 0

for ww in range(np.size(end, axis=0)):
    maze[end[ww, 0]] = 0
    maze[end[ww, 1]] = 0

obs = np.where(maze==1)
obsx = np.asarray(obs[1])
obsy = np.asarray(obs[0])

ps = []
pay = np.zeros((np.size(start, axis=0), np.size(end, axis=0)))

for jj in range(np.size(start, axis=0)):
    for uu in range(np.size(end, axis=0)):
        path_list = astar.search(maze, cost, start[jj, :], end[uu, :])
        path = np.asarray(path_list)
        ps.append(path)
        pay[jj, uu] = np.max(path)

pick = np.zeros((np.size(pay, axis=0), 1))
scale = 0

for yy in range(np.size(pay, axis=0)):
    # pick[yy, 0] = np.max(np.asarray(np.where(pay[yy, :] == np.min(pay[yy, :]))))
    pick[yy, 0] = np.max(np.asarray(np.where(pay[yy, :] == np.min(pay[yy, :]))))
    pay[:, int(pick[yy, 0])] = 3e10
    pick[yy, 0] = pick[yy, 0] + scale*np.size(start, axis=0)
    scale = scale+1
    
pa = []
# pa.append(ps[int(pick[0])])
# for oo in range(np.size(pick)-1):
#     pa.append(ps[int(int(np.size(start, axis=0))+pick[oo+1])])
    
for oo in range(np.size(pick)):
    pa.append(ps[int(pick[oo])])

targlist = []

xt = np.zeros((np.size(end, axis=0)))
yt = np.zeros((np.size(end, axis=0)))
wpt = np.zeros((np.size(start, axis=0)))

for tt in range(np.size(pick)):
    path = pa[tt]
    idx = np.zeros((np.max(path), 2))
    targ1 = np.zeros((np.max(path), 2))
    pt = 0
    for qq in range(np.max(path)):
        where = np.where(path==pt)
        idx[qq, 0] = where[1]
        idx[qq, 1] = where[0]
        pt = pt+1
        targ1[qq, 0] = xm[np.int(idx[qq, 0])]
        targ1[qq, 1] = ym[np.int(idx[qq, 1])]
    targlist.append(targ1)
    xt[tt] = targ1[0, 0]
    yt[tt] = targ1[0, 1]
    wpt[tt] = 0

xy = []
dxy = []
xa = np.zeros((np.size(pick), 1))
ya = np.zeros((np.size(pick), 1))
xa_i = np.zeros((np.size(pick), 1))
ya_i = np.zeros((np.size(pick), 1))
d2d_i = np.zeros((np.size(pick), 1))
Hdes_i = np.zeros((np.size(pick), 1))

for pp in range(np.size(pick)):
    xy.append(np.zeros((2, maxiter)))
    xy[pp][0, 0] = xm[start[pp, 1]]
    xy[pp][1, 0] = ym[start[pp, 0]]
    xa[pp] = xm[start[pp, 1]]
    ya[pp] = ym[start[pp, 0]]
    xa_i[pp] = xa[pp]
    ya_i[pp] = ya[pp]
    [d2d_i[pp], Hdes_i[pp]] = guide.guidance(xa[pp], ya[pp], xt[pp], yt[pp])
    dxy.append(np.zeros((1, maxiter)))
    dxy[pp][0, 0] = d2d_i[pp]

#%% Simulate LQR
x = []
xdes = []
xs = []
leng = []
d2d = np.zeros((np.size(pick), 1))
Hdes = np.zeros((np.size(pick), 1))

for aa in range(np.size(pick)):
    targ = targlist[aa]
    x.append(np.zeros([statedim, 1]))
    x[aa][0] = 3.0
    x[aa][4] = np.deg2rad(Hdes_i[aa])
    v = x[aa][0, 0]
    psi = x[aa][4, 0]
    u = 10.0
    xdes.append(np.zeros([statedim, 1]))
    xdes[aa][4, 0] = Hdes_i[aa]
    xs.append(np.zeros((statedim, maxiter)))
    xs[aa][:, 0] = np.ndarray.flatten(x[aa])
    
    for ii in range(maxiter):
        if ii % mod == 0:
            [d2d[aa], Hdes[aa]] = guide.guidance(xa[aa], ya[aa], xt[aa], yt[aa])
            dxy[aa][0, ii] = d2d[aa]
            xdes[aa][4] = Hdes[aa]
            x[aa] = Fr@x[aa]+Gr@xdes[aa]
            xs[aa][:, ii] = np.ndarray.flatten(x[aa])
            v = x[aa][0, 0]
            psi = x[aa][4, 0]
            [xa[aa], ya[aa]] = guide.velprop(xa[aa], ya[aa], u, v, psi, dt)
            xy[aa][0, ii] = xa[aa]
            xy[aa][1, ii] = ya[aa]
            if d2d[aa] < 100:
                wpt[aa] = wpt[aa]+1
                if wpt[aa] == np.size(targ, axis=0):
                    np.disp('Im Finished!')
                    leng.append(ii)
                    break
                else:
                    xt[aa] = targ[int(wpt[aa]), 0]
                    yt[aa] = targ[int(wpt[aa]), 1]
                    [d2d[aa], Hdes[aa]] = guide.guidance(xa[aa], ya[aa], xt[aa], yt[aa])
                    dxy[aa][0, ii] = d2d[aa]
                    xdes[aa][4] = Hdes[aa]
        else:
            x[aa] = Fr@x[aa]+Gr@xdes[aa]
            xs[aa][:, ii] = np.ndarray.flatten(x[aa])
            v = x[aa][0, 0]
            [xa[aa], ya[aa]] = guide.velprop(xa[aa], ya[aa], u, v, psi, dt)
            xy[aa][0, ii] = xa[aa]
            xy[aa][1, ii] = ya[aa]

#%% Plots

for gg in range(np.size(pick)):
    plt.plot(xy[gg][0, :leng[gg]], xy[gg][1, :leng[gg]], label='Aircraft path', linewidth=3)
    plt.scatter(targlist[gg][:, 0], targlist[gg][:, 1], s=5, label='Waypoints', color='magenta')
    plt.plot(xm[start[gg, 1]], ym[start[gg, 0]], 'ro')
    plt.plot(xm[end[gg, 1]], ym[end[gg, 0]], 'rx')
plt.scatter(xm[obsx], ym[obsy], s=30, label='Obstacles', color='black')
plt.xlim(-0.05*mapsize, mapsize+mapsize*0.05)
plt.ylim(-0.05*mapsize, mapsize+mapsize*0.05)
# plt.legend()
plt.show()

# plt.plot(t, np.ndarray.flatten(dxy))
# plt.show()