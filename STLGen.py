# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:18:02 2022

@author: nbagz
"""
import sys


from HullParameterization import Hull_Parameterization as HP

import numpy as np

import csv

from tqdm import tqdm


#Open the Design Vector csv

for j in range(1,4):
    path = './Constrained_Randomized_Set_' + str(j) + '/'
    filename = 'Input_Vectors.csv'
    Vec = []
    with open(path + filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            Vec.append(row)

    #Save as a np.float array
    Vec = np.array(Vec)
    DesVec = Vec.astype(np.float64())
    
    print('Meshing STLs for ' + path)
    

    #loop thru to make point cloud files
    for i in tqdm(range(0,len(DesVec)):
    
    
        hull = HP(DesVec[i])
    
            
        #PC = hull.gen_pointCloud(NUM_WL = 50, PointsPerWL = 500)
    
        strpath =  path + 'stl/Hull_mesh_' + str(i) # + '.stl'
    
        mesh = hull.gen_stl(NUM_WL= 100, PointsPerWL = 800,  namepath = strpath)
    

    
