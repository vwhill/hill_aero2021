# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:24:44 2020

@author: vince
"""
# -*- coding: utf-8 -*-

#%% Modules

import astar_single as astar
import guideclass as guide
import numpy as np
import matplotlib.pyplot as plt
import gncpy.filters as filters
import gasur.swarm_estimator.tracker as track
from gasur.utilities.distributions import GaussianMixture

#%% Initialize

start = np.array([[0, 0],
                    # [10, 0], 
                    [20, 0],
                    # [30, 0],
                    # [40, 0],
                   [50, 0]])

end = np.array([[5, 50],
                # [25, 25],
                [15, 50],
                # [25, 50],
                # [35, 50],
                [45, 50]])

dt = 0.01
dyn = guide.LateralFixedWing(dt)

statedim = dyn.statedim
indim = dyn.indim

sysr = guide.LQR(dt, dyn.F, dyn.G, statedim, indim)
Fr = sysr.Fr
Gr = sysr.Gr

Ai = np.array([[0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0]])

Bi = np.array([[0.0, 0.0],
               [0.0, 0.0],
               [1.0, 0.0],
               [0.0, 1.0]])

Ci = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])

[Fi, Gi] = guide.discretize(1.0, Ai, Bi, Ci, np.zeros((2, 2)))

wi = 10.0*np.ones((4, 1))
wi[2, 0] = 100.0
wi[3, 0] = 100.0
vi = 10.0*np.ones((2, 1))

Q = np.zeros((4, 4))
R = np.zeros((2, 2))

for xx in range(np.size(wi, axis=0)):
    Q[xx, xx] = wi[xx]
for xx2 in range(np.size(vi, axis=0)):
    R[xx2, xx2] = vi[xx2]

kf = filters.KalmanFilter()
kf.set_proc_noise(mat=Q)
kf.meas_noise = R
kf.set_state_mat(mat=Fi)
kf.set_input_mat(mat=Gi)
kf.set_meas_mat(mat=Ci)

maxtime = 5000
maxiter = np.int(maxtime/dt)
t = np.linspace(0, maxtime, maxiter)

#%% Create Mission Plan

a = 51
b = 51
mapsize = 10000
[mesh, xm, ym] = astar.meshgen(a, b, mapsize)

[maze, cost] = astar.mazegen(a, b)

for hh in range(np.size(start, axis=0)):
    maze[start[hh, 0], start[hh, 1]] = 0

for ww in range(np.size(end, axis=0)):
    maze[end[hh, 0], end[hh, 1]] = 0

obs = np.where(maze==1)
obsx = np.asarray(obs[1])
obsy = np.asarray(obs[0])

ps = []
targlist = []
pay = np.zeros((np.size(start, axis=0), np.size(end, axis=0)))
hun = []

for jj in range(np.size(start, axis=0)):
    for uu in range(np.size(end, axis=0)):
        path_list = astar.search(maze, cost, start[jj, :], end[uu, :])
        path = np.asarray(path_list)
        ps.append(path)
        idx = np.zeros((np.max(path), 2))
        targ1 = np.zeros((np.max(path), 2))
        pt = 0
        hun = 0
        for qq in range(np.max(path)):
            where = np.where(path==pt)
            idx[qq, 0] = where[1]
            idx[qq, 1] = where[0]
            pt = pt+1
            targ1[qq, 0] = xm[np.int(idx[qq, 0])]
            targ1[qq, 1] = ym[np.int(idx[qq, 1])]
        for vv in range(np.size(targ1, axis=0)-1):
            hun = hun+np.sqrt((targ1[vv+1, 0]-targ1[vv, 0])**2+(targ1[vv+1, 1]-targ1[vv, 1])**2)
        targlist.append(targ1)
        pay[jj, uu] = hun

pick = np.zeros((np.size(pay, axis=0), 1))
scale = 0

for yy in range(np.size(pay, axis=0)):
    # pick[yy, 0] = np.max(np.asarray(np.where(pay[yy, :] == np.min(pay[yy, :]))))
    pick[yy, 0] = np.max(np.asarray(np.where(pay[yy, :] == np.min(pay[yy, :]))))
    pay[:, int(pick[yy, 0])] = 3e10
    pick[yy, 0] = pick[yy, 0] + scale*np.size(start, axis=0)
    scale = scale+1
    
pa = []
waypoints = []
xt = np.zeros((np.size(end, axis=0)))
yt = np.zeros((np.size(end, axis=0)))
wpt = np.zeros((np.size(start, axis=0)))

for oo in range(np.size(pick)):
    pa.append(ps[int(pick[oo])])
    waypoints.append(targlist[int(pick[oo])])
    xt[oo] = waypoints[oo][0, 0]
    yt[oo] = waypoints[oo][0, 1]
    wpt[oo] = 0

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

#%% Initialize Sim

x = []
y = []
ys = []
xi = []
xdes = []
xs = []
cov = []
gm = []
birth = []
leng = np.zeros((np.size(pick), 1))
d2d = np.zeros((np.size(pick), 1))
Hdes = np.zeros((np.size(pick), 1))
u = 10.0
done = 0
mod = 100

for zz in range(np.size(pick)):
    x.append(np.zeros([statedim, 1]))
    y.append(np.zeros([statedim, 1]))
    # xi.append(np.zeros([4, 1]))
    xdes.append(np.zeros([statedim, 1]))
    xs.append(np.zeros((statedim, maxiter)))
    cov.append(np.zeros((2, 2)))
    x[zz][0] = 3.0
    x[zz][4] = 0
    xdes[zz][4, 0] = Hdes_i[zz]
    xs[zz][:, 0] = np.ndarray.flatten(x[zz])
    xi.append(np.array([[float(xa[zz])], [float(ya[zz])], [0.0], [0.0]]))
    # x0 = xi[zz] + 100.0*np.random.rand(2, 1)
    # cov[zz] = (xi[zz]-x0)@(xi[zz]-x0).T
    cov[zz] = 100.0*np.eye(4)
    cov[zz][2, 2] = 20.0
    cov[zz][3, 3] = 20.0
    gm.append(GaussianMixture(means=[xi[zz].copy()], covariances=[cov[zz].copy()], weights=[1]))
    birth.append((gm[zz], 0.03))

rfs = track.GeneralizedLabeledMultiBernoulli()
rfs.prob_detection = 0.99
rfs.prob_survive = 0.99
rfs.req_births = np.size(start, axis=0)+1
rfs.birth_terms = birth
rfs.req_surv = 3000
rfs.req_upd = 3000
rfs.inv_chi2_gate = 32.2361913029694
rfs.gating_on = False
rfs.clutter_rate = 0.0001
rfs.clutter_den = 1 / (np.pi * 2000)
rfs.filter = kf

#%% Simulate

count = 0

for ii in range(maxiter):
    asdf = []
    if done == np.size(pick):
        break
    for aa in range(np.size(pick)): # dynamics
        targ = waypoints[aa]
        if wpt[aa] == np.size(targ, axis=0):
            continue
        x[aa] = Fr@x[aa]+Gr@xdes[aa]
        xs[aa][:, ii] = np.ndarray.flatten(x[aa])
        [xa[aa], ya[aa]] = guide.velprop(xa[aa], ya[aa], u, x[aa][0, 0], x[aa][4, 0], dt)
        xy[aa][0, ii] = xa[aa]
        xy[aa][1, ii] = ya[aa]
        y[aa] = np.array([xa[aa], ya[aa]])
        asdf.append(np.array([[xa[aa]], [ya[aa]], [0.0], [0.0]], dtype=np.float32))
    ys.append(asdf)

    if ii % mod == 0:
        rfs.predict(time_step=count) # GLMB
        rfs.correct(meas=y)
        rfs.prune()
        rfs.cap()
        rfs.extract_states()
        count = count+1
        
        for bb in range(np.size(pick)):   # guidance
            targ = waypoints[bb]
            if wpt[bb] == np.size(targ, axis=0):
                continue
            [d2d[bb], Hdes[bb]] = guide.guidance(xa[bb], ya[bb], xt[bb], yt[bb])
            dxy[bb][0, ii] = d2d[bb]
            xdes[bb][4] = Hdes[bb]
            if d2d[bb] < 50:
                wpt[bb] = wpt[bb]+1
                if wpt[bb] == np.size(targ, axis=0):
                    np.disp('Im Finished!')
                    done = done+1
                    leng[bb] = ii
                    break
                else:
                    xt[bb] = targ[int(wpt[bb]), 0]
                    yt[bb] = targ[int(wpt[bb]), 1]
                    [d2d[bb], Hdes[bb]] = guide.guidance(xa[bb], ya[bb], xt[bb], yt[bb])
                    dxy[bb][0, ii] = d2d[bb]
                    xdes[bb][4] = Hdes[bb]

#%% Plots

qwer=rfs.plot_states_labels([0, 1])

plt.figure()
for gg in range(np.size(pick)):
    plt.plot(xy[gg][0, :int(leng[gg])], xy[gg][1, :int(leng[gg])], label='Aircraft path', linewidth=3)
    plt.scatter(waypoints[gg][:, 0], waypoints[gg][:, 1], s=5, label='Waypoints', color='magenta')
    plt.plot(xm[start[gg, 1]], ym[start[gg, 0]], 'ro')
    plt.plot(xm[end[gg, 1]], ym[end[gg, 0]], 'rx')
plt.scatter(xm[obsx], ym[obsy], s=30, label='Obstacles', color='black')
qwer.axes[0].set_xlim((-0.05*mapsize, mapsize+mapsize*0.05))
qwer.axes[0].set_ylim((-0.05*mapsize, mapsize+mapsize*0.05))
# plt.xlim(-0.05*mapsize, mapsize+mapsize*0.05)
# plt.ylim(-0.05*mapsize, mapsize+mapsize*0.05)
# plt.legend()
plt.show()

rfs.plot_card_dist()