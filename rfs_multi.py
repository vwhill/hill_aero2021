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
from datetime import datetime
from copy import deepcopy

rng = np.random.default_rng(13)

now = datetime.now()
start_time = now.strftime("%H:%M:%S")
print('Start Time = ', start_time)

#%% Initialize

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

maxtime = 1000
maxiter = np.int(maxtime/dt)
t = np.linspace(0, maxtime, maxiter)

#%% Create Mission Plan

start = np.array([[5, 0],
                    # [20, 0],
                  [45, 0]])

end = np.array([[5, 45],
                # [25, 45],
                [45, 45]])

init_start = start.copy()
init_end = end.copy()

a = 51
b = 51
mapsize = 2000
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
        path = np.asarray(path_list).copy()
        ps.append(path)
        idx = np.zeros((np.max(path), 2))
        targ1 = np.zeros((np.max(path), 2))
        pt = 0
        hun = 0
        for qq in range(np.max(path)):
            where = np.where(path==pt)
            idx[qq, 0] = where[1].copy()
            idx[qq, 1] = where[0].copy()
            pt = pt+1
            targ1[qq, 0] = xm[np.int(idx[qq, 0])].copy()
            targ1[qq, 1] = ym[np.int(idx[qq, 1])].copy()
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
    xt[oo] = waypoints[oo][0, 0].copy()
    yt[oo] = waypoints[oo][0, 1].copy()
    wpt[oo] = 0

# init_waypoints = waypoints.copy()

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
    xy[pp][0, 0] = xm[start[pp, 1]].copy()
    xy[pp][1, 0] = ym[start[pp, 0]].copy()
    xa[pp] = xm[start[pp, 1]].copy()
    ya[pp] = ym[start[pp, 0]].copy()
    xa_i[pp] = xa[pp].copy()
    ya_i[pp] = ya[pp].copy()
    [d2d_i[pp], Hdes_i[pp]] = guide.guidance(xa[pp], ya[pp], xt[pp], yt[pp])
    dxy.append(np.zeros((1, maxiter)))
    dxy[pp][0, 0] = d2d_i[pp].copy()

#%% Initialize Sim

x = []
y = []
ys = []
xi = []
xdes = []
xs = []
gm = []
birth = []
agestate = []
agemeas = []
targstate = []
targmeas = []
targsave = []
cov = np.zeros((2, 2))
cov = 100.0*np.eye(4)
cov[2, 2] = 20.0
cov[3, 3] = 20.0
leng = np.zeros((np.size(pick), 1))
d2d = np.zeros((np.size(pick), 1))
Hdes = np.zeros((np.size(pick), 1))
u = 5.0
done = 0
mod = 100

for zz in range(np.size(pick)):
    x.append(np.zeros([statedim, 1]))
    y.append(np.zeros([statedim, 1]))
    xdes.append(np.zeros([statedim, 1]))
    xs.append(np.zeros((statedim, maxiter)))
    # cov.append(np.zeros((2, 2)))
    agestate.append(np.zeros((4, 1)))
    agemeas.append(np.zeros((2, 1)))
    targstate.append(np.zeros((4, 1)))
    targmeas.append(np.zeros((2, 1)))
    targsave.append(np.zeros((2, maxiter)))
    x[zz][0] = 3.0
    x[zz][4] = 0
    xdes[zz][4, 0] = Hdes_i[zz].copy()
    xs[zz][:, 0] = np.ndarray.flatten(x[zz]).copy()
    targstate[zz][0] = xm[end[zz, 1]].copy()
    targstate[zz][1] = ym[end[zz, 0]].copy()
    xi.append(np.array([[float(xa[zz])], [float(ya[zz])], [0.0], [0.0]]))
    xi.append(np.array([[xm[end[zz, 1]]], [ym[end[zz, 0]]], [0.0], [0.0]]))

    
for cc in range(np.size(start, axis=0)+np.size(end, axis=0)):
    gm.append(GaussianMixture(means=[xi[cc].copy()], covariances=[cov.copy()], weights=[1]))
    birth.append((gm[cc], 0.03))

a_inds = [ii for ii in range(0, len(birth), 2)]
t_inds = [ii for ii in range(1, len(birth), 2)]
gma = [birth[ii] for ii in a_inds]
gmt = [birth[ii] for ii in t_inds]
rfs = track.GeneralizedLabeledMultiBernoulli()
rfs.prob_detection = 0.99
rfs.prob_survive = 0.99
rfs.req_births = np.size(start, axis=0)+1
rfs.birth_terms = gma
rfs.req_surv = 3000
rfs.req_upd = 3000
rfs.inv_chi2_gate = 32.2361913029694
rfs.gating_on = False
rfs.clutter_rate = 0.0001
rfs.clutter_den = 1 / (np.pi * 2000)
rfs.filter = kf

rfs2 = deepcopy(rfs)
rfs2.birth_terms = gmt
rfs.req_births = np.size(end, axis=0)+1

#%% Simulate

targstate[0][2] = -0.01 # prescribe target motion
targstate[0][3] = 0.03
targstate[1][2] = -0.01
targstate[1][3] = -0.03

# targstate[0][2] = -0.01 # prescribe target motion
# targstate[0][3] = 0.02 
# targstate[1][2] = 0.0
# targstate[1][3] = -0.01
# targstate[2][2] = -0.02
# targstate[2][3] = 0.0

count = 0
count1 = 0

change = np.zeros(np.size(end, axis=0))
x_end = np.zeros(np.size(end, axis=0))
y_end = np.zeros(np.size(end, axis=0))
x_start = np.zeros(np.size(start, axis=0))
y_start = np.zeros(np.size(start, axis=0))

for ii in range(0, maxiter):
    truestate = []
    y = []
    if done == np.size(pick):
        break
    for aa in range(np.size(pick)): # agent dynamics
        targ = waypoints[aa].copy()
        if wpt[aa] == np.size(targ, axis=0):
            continue
        x[aa] = Fr@x[aa]+Gr@xdes[aa]
        xs[aa][:, ii] = np.ndarray.flatten(x[aa])
        [xa[aa], ya[aa]] = guide.velprop(xa[aa], ya[aa], u, x[aa][0, 0], x[aa][4, 0], dt)
        xy[aa][0, ii] = xa[aa]
        xy[aa][1, ii] = ya[aa]
        agestate[aa] = np.array([xa[aa], ya[aa], [0.0], [0.0]])
        agemeas[aa] = np.array([xa[aa], ya[aa]])
        y.append(np.array([xa[aa], ya[aa]]))
        truestate.append(np.array([[xa[aa]], [ya[aa]], [0.0], [0.0]], dtype=np.float32))
    ys.append(truestate)
    
    for ee in range(np.size(end, axis=0)): # target dynamics
        targ = waypoints[ee].copy()
        if wpt[ee] == np.size(targ, axis=0):
            continue
        targstate[ee] = Fi@targstate[ee]+Gi@np.zeros((2, 1))
        targmeas[ee] = Ci@targstate[ee]
        y.append(targmeas[ee])
        targsave[ee][0, ii] = float(targmeas[ee][0])
        targsave[ee][1, ii] = float(targmeas[ee][1])
    
    # clutter goes here
    
    for bb in range(np.size(pick)):   # guidance
        targ = waypoints[bb].copy()
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
                xt[bb] = targ[int(wpt[bb]), 0].copy()
                yt[bb] = targ[int(wpt[bb]), 1].copy()
                [d2d[bb], Hdes[bb]] = guide.guidance(xa[bb], ya[bb], xt[bb], yt[bb])
                dxy[bb][0, ii] = d2d[bb]
                xdes[bb][4] = Hdes[bb]
        
    if ii % mod == 0:
        rfs.predict(time_step=count) # GLMB
        rfs.correct(meas=y)
        rfs.prune()
        rfs.cap()
        rfs.extract_states()
        
        rfs2.predict(time_step=count) # GLMB
        rfs2.correct(meas=y)
        rfs2.prune()
        rfs2.cap()
        rfs2.extract_states()
        
        a_state = []
        t_state = []
        if len(rfs.states[-1]) > 1:
            a_state = []
            for jj in range(len(rfs.states[-1])):
                a_state.append(np.array([rfs.states[-1][jj][0], rfs.states[-1][jj][1]]))
        else:
            a_state = np.array([rfs.states[-1][0][0], rfs.states[-1][0][1]])
        if len(rfs2.states[-1]) > 1:
            t_state = []
            for jj in range(len(rfs2.states[-1])):
                t_state.append(np.array([rfs2.states[-1][jj][0], rfs2.states[-1][jj][1]]))
        else:
            t_state = np.array([rfs2.states[-1][0][0], rfs2.states[-1][0][1]])
        
        count = count+1
        
        if count < 3:
            continue
        
        a_norm = []
        a_assoc = []
        a_state_fixed = []
        for qq in range(len(a_state)):
            for gg in range(len(agemeas)):
                a_norm.append(np.linalg.norm(np.abs(agemeas[gg]-a_state[qq])))
        for ff in range(0, len(a_state)):
            a_assoc.append(np.argmin(a_norm[(ff*len(pick)):(2*ff+len(pick))]))
        for pp in range(len(a_state)):
            a_state_fixed.append(a_state[a_assoc[pp]])
        a_state = a_state_fixed.copy()
        
        t_norm = []
        t_assoc = []
        t_state_fixed = []
        for qq in range(len(t_state)):
            for gg in range(len(targmeas)):
                t_norm.append(np.linalg.norm(np.abs(targmeas[gg]-t_state[qq])))
        for ff in range(0, len(t_state)):
            t_assoc.append(np.argmin(t_norm[(ff*len(pick)):(2*ff+len(pick))]))
        for pp in range(len(t_state)):
            t_state_fixed.append(t_state[t_assoc[pp]])
        t_state = t_state_fixed.copy()
        
        if ii % (mod*10) == 0:
        # if ii % mod == 0:
            for i1 in range(np.size(end, axis=0)):
                change[i1] = np.sqrt((float(t_state[i1][0])-xm[end[i1, 1]])**2+(float(t_state[i1][1])-ym[end[i1, 0]])**2)
                if change[i1] > 50:
                    y_end[i1] = np.argmin(np.abs(ym-float(t_state[i1][1])))
                    x_end[i1] = np.argmin(np.abs(xm-float(t_state[i1][0])))
                    end[i1, 0] = y_end[i1].copy()
                    end[i1, 1] = x_end[i1].copy()
                    y_start[i1] = np.argmin(np.abs(ym-float(a_state[i1][1])))
                    x_start[i1] = np.argmin(np.abs(xm-float(a_state[i1][0])))
                    start[i1, 0] = y_start[i1].copy()
                    start[i1, 1] = x_start[i1].copy()
                    ps = []
                    targlist = []
                    pay = np.zeros((np.size(start, axis=0), np.size(end, axis=0)))
                    hun = []
                    for jj in range(np.size(start, axis=0)):
                        for uu in range(np.size(end, axis=0)):
                            path_list = astar.search(maze, cost, start[jj, :], end[uu, :])
                            path = np.asarray(path_list).copy()
                            ps.append(path)
                            idx = np.zeros((np.max(path), 2))
                            targ1 = np.zeros((np.max(path), 2))
                            pt = 0
                            hun = 0
                            for qq in range(np.max(path)):
                                where = np.where(path==pt)
                                idx[qq, 0] = where[1].copy()
                                idx[qq, 1] = where[0].copy()
                                pt = pt+1
                                targ1[qq, 0] = xm[np.int(idx[qq, 0])].copy()
                                targ1[qq, 1] = ym[np.int(idx[qq, 1])].copy()
                            for vv in range(np.size(targ1, axis=0)-1):
                                hun = hun+np.sqrt((targ1[vv+1, 0]-targ1[vv, 0])**2+(targ1[vv+1, 1]-targ1[vv, 1])**2)
                            targlist.append(targ1)
                            pay[jj, uu] = hun
                    
                    pick = np.zeros((np.size(pay, axis=0), 1))
                    scale = 0
                    
                    for yy in range(np.size(pay, axis=0)):
                        pick[yy, 0] = np.max(np.asarray(np.where(pay[yy, :] == np.min(pay[yy, :]))))
                        pay[:, int(pick[yy, 0])] = 3e10
                        pick[yy, 0] = pick[yy, 0] + scale*np.size(start, axis=0)
                        scale = scale+1
                    pa = []
                    waypoints = []

                    for oo in range(np.size(pick)):
                        pa.append(ps[int(pick[oo])])
                        waypoints.append(targlist[int(pick[oo])])
                        xt[oo] = waypoints[oo][0, 0].copy()
                        yt[oo] = waypoints[oo][0, 1].copy()
                        wpt[oo] = 0
            count1 = count1+1

#%% Plots

for ll in range(np.size(pick)):
    if leng[ll] == 0:
        leng[ll] = ii

# rfs.plot_card_dist()
# plt.savefig('card_dist.eps', format='eps')

rfsplot=rfs.plot_states_labels([0, 1])
rfs2.plot_states_labels([0, 1], f_hndl=rfsplot)
plt.savefig('rfsplot.eps', format='eps')

plt.figure()
for gg in range(np.size(pick)):
    plt.plot(xy[gg][0, :int(leng[gg])], xy[gg][1, :int(leng[gg])], label='Aircraft path', linewidth=3)
    plt.plot(targsave[gg][0, :int(leng[gg])], targsave[gg][1, :int(leng[gg])], label='Target path', linewidth=2)
    # plt.scatter(init_waypoints[gg][:, 0], init_waypoints[gg][:, 1], s=5, label='Initial Waypoints', color='blue')
    plt.scatter(waypoints[gg][:, 0], waypoints[gg][:, 1], s=5, label='Current Waypoints', color='magenta')
    plt.plot(xm[init_start[gg, 1]], ym[init_start[gg, 0]], 'go', label='Initial Start')
    plt.plot(xm[init_end[gg, 1]], ym[init_end[gg, 0]], 'gx', label='Initial End')
    # plt.plot(xm[start[gg, 1]], ym[start[gg, 0]], 'ro', label='Updated Start')
    plt.plot(xm[end[gg, 1]], ym[end[gg, 0]], 'rx', label='Updated End')
plt.scatter(xm[obsx], ym[obsy], s=30, label='Obstacles', color='black')
rfsplot.axes[0].set_xlim((-0.05*mapsize, mapsize+mapsize*0.2))
rfsplot.axes[0].set_ylim((-0.05*mapsize, mapsize+mapsize*0.2))
plt.xlabel('x position')
plt.ylabel('y position')
# plt.xlim(-0.05*mapsize, mapsize+mapsize*0.05)
# plt.ylim(-0.05*mapsize, mapsize+mapsize*0.05)
# plt.legend()
plt.savefig('aircraftplot.eps', format='eps')
plt.show()

now = datetime.now()
end_time = now.strftime("%H:%M:%S")
print('End Time = ', end_time)

#%% Clutter code

# num_clutt = rng.poisson(clutter_rate)
# for ff in range(num_clutt):
#     m = pos_bnds[:, [0]] + (pos_bnds[:, [1]] - pos_bnds[:, [0]]) \ rng.random((2, 1))
#     meas.append(m)