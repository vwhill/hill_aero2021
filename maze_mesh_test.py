# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 08:18:45 2020

@author: vince
"""

import numpy as np

a = 100
b = 100
mapsize = 3000 # map square edge length

x=np.linspace(0,mapsize,a)
y=np.linspace(0,mapsize,b)

mesh = np.zeros((a, b))

mesh[:, 0] = y
mesh[0, :] = x

mesh[a-1, :] = np.linspace(mapsize, np.sqrt(2*mapsize**2), a)
mesh[:, b-1] = np.linspace(mapsize, np.sqrt(2*mapsize**2), b)

for i in range(1,a):
    mesh[i, :] = np.linspace(mesh[i, 0], mesh[i, b-1], b)