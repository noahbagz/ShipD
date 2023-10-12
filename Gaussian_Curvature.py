#This file contains all of the functions to calculate the gaussian curvature for the ship Hulls


from HullParameterization import Hull_Parameterization as HP

import numpy as np

def SurfArea(X,Z,Y):
    #generate an array of surface areas to evaluate the curvature of a hull
    
    A = np.zeros((len(X)-2,len(Z)-2))
    
    C = np.zeros((4,))
    nx = len(X)-2
    nz = len(Z)-2
    
    for i in range(1,len(X)-1):
        
        for j in range(1,len(Z)-1):
            
            C = np.zeros((4,)) + 0.5
            
            if i == 1:
                C[0] = 1.0
            if i == nx:
                C[1] = 1.0
            if j == 1:
                C[2] = 1.0
            if j == nz:
                C[3] = 1.0
            
            L1 = C[0] * np.sqrt((X[i] - X[i-1])**2.0 + (Y[i,j] - Y[i-1,j])**2.0) + C[1] * np.sqrt((X[i] - X[i+1])**2.0 + (Y[i,j] - Y[i+1,j])**2.0)
            
            L2 = C[2] * np.sqrt((Z[j] - Z[j-1])**2.0 + (Y[i,j] - Y[i,j-1])**2.0) + C[3] * np.sqrt((Z[j] - Z[j+1])**2.0 + (Y[i,j] - Y[i,j+1])**2.0)
            
            
            A[i-1,j-1] = L1*L2
    
    return A
            

def GaussianCurvature(x):
    #create a hull and a structured PC (use PC for michelle flow)
    hull = HP(x)
    
    X,Z,Y,WL = hull.gen_PC_for_Cw(hull.Dd, NUM_WL=501, PointsPerWL = 501)
    
    A = SurfArea(X,Z,Y) #generate grid of small delta_areas
    
    SA = np.sum(A) #total surface area
    
    hx = X[1] - X[0]
    hz = Z[1] - Z[0]
    
    GK = 0
    
    for i in range(1,len(X)-1):
        
        for j in range(1,len(Z)-1):
            # Calcuate the Curvature in the XY plane (K1) and the ZY plane (K2)
            #K is solved using finite difference: K = Y''/(1+Y'^2)^3/2
            
            K1 = ((Y[i-1,j] - 2*Y[i,j] + Y[i+1,j])/hx**2.0)/((1 + ((Y[i+1,j]-Y[i-1,j])/(2*hx))**2.0)**1.5)
            
            K2 = ((Y[i,j-1] - 2*Y[i,j] + Y[i,j+1])/hz**2.0)/((1 + ((Y[i,j+1]-Y[i,j-1])/(2*hz))**2.0)**1.5)
            
            
            # moment of curv
            GK = GK + abs(K1*K2*A[i-1,j-1])
    
    return (x[0]**2.0)*GK/SA #weighted average of gaussian curvature normalized to LOA^2
        

    