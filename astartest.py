# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 08:39:41 2020

@author: vince
"""

import numpy as np
import astar_single as astar

a = 100
b = 100
mapsize = 5000
[mesh, xm, ym] = astar.meshgen(a, b, mapsize)

[maze, start, end, cost, obs] = astar.mazegen(a, b)

obsx = np.asarray(obs[0])
obsy = np.asarray(obs[1])

path_list = astar.search(maze, cost, start, end)
path = np.asarray(path_list)

pathidx = np.where(path>0)
pathidx_x = np.asarray(pathidx[1])
pathidx_y = np.asarray(pathidx[0])
targ = np.zeros((np.size(pathidx[0]), 2))

for jj in range(0, np.size(pathidx_x)):
    targ[jj, 0] = xm[pathidx_x[jj]]
    targ[jj, 1] = ym[pathidx_y[jj]]