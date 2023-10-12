# This file contains all of the functions to calculate the MaxBox Metric for the ship Hulls

from HullParameterization import Hull_Parameterization as HP

import numpy as np
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead

from pymoo.core.problem import ElementwiseProblem

import csv
from tqdm import tqdm




# Set Up Nelder-Mead 

def BoxVol(inputs):
    '''
    inputs = [Xf,Xa,h,w] = [forward start of box, aft end of box, height of box, half-beam of box]
    
    Vol = 2*w*h*(Xa-Xf)
    '''

    return 2*inputs[3]*inputs[2]*(inputs[1]-inputs[0])

def BoxCons(inputs,hull):
    '''
    inputs = [Xf,Xa,h,w] = [forward start of box, aft end of box, height of box, half-beam of box]
    
    hull is used to calculate the positioning of the corners of the box
    
    '''
    
    corners = np.zeros((4,))
    depths = [hull.Dd, hull.Dd-inputs[2]]
    
    #Calc position of delta_bow and delta_stern at Deck and at Dd-H

    
    Pos = np.array([[hull.delta_bow(hull.Dd), hull.delta_stern(hull.Dd)],
           [hull.delta_bow(hull.Dd-inputs[2]), hull.delta_stern(hull.Dd-inputs[2])]])
    
    for i in range(0,2):
        # i denotes forward or aft corner
        for j in range(0,2):
            # j denotes upper (Dd) or lower corner (Dd-h)
                       
            #First check if the input is in the stern taper:
            if inputs[i] > Pos[j,1]:
                    
                if inputs[i] > hull.stern_profile(depths[j]):
                    #assign corner to zero if its outside the stern profile
                    y = 0.0
                
                else:
                    #assign it to the half beam of the stern at the x position otherwise
                    y = hull.halfBeam_Stern([inputs[i]],hull.solve_waterline_stern(depths[j]))
             
            #next check if the position is in the bow profile
            elif inputs[i] < Pos[j,0]:
                
                #check if the position is fwd of the bow profile
                if inputs[i] < hull.bow_profile(depths[j]):
                    y = 0.0
                else:
                    #assign it to the halfbeam at that position otherwise
                    y = hull.halfBeam_Bow([inputs[i]],hull.solve_waterline_bow(depths[j]))
            
            else:
                #assign the corner position to the half beam of the mid body if the x position is in the parallel mid body
                y = hull.halfBeam_MidBody(depths[j])
            
            corners[2*i+j] = y
            
    return inputs[3] - corners



def PC_BoxCon(points, box_inputs,Dd):
    # points: numpy array of shape (N, 3)
    # box_dims: list of length [xa,xf,h,w]
    # Get the minimum and maximum values for each dimension of the box
    box_min = np.array([box_inputs[0], 0, Dd-box_inputs[2]])
    box_max = np.array([box_inputs[1], box_inputs[3], Dd])

    # Check if any of the points are within the box
    in_box = np.logical_and(np.all(points >= box_min, axis=1),
                            np.all(points <= box_max, axis=1))

    return np.sum(in_box)
        
        

        
class BoxOpt(ElementwiseProblem):
    
    # When intializing, set the mechanism size and target curve
    def __init__(self, x):
        
        #set up hull, bounds, and initial guess for the problem
        self.hull = HP(x)
        
        self.PC = self.hull.gen_MeshGridPointCloud(NUM_WL = 201, PointsPerLOA = 501, Z = [], X = [], bit_GridOrList = 0)
    
        #self.x0 = [self.hull.Lb,self.hull.LOA-self.hull.Ls,0.9*self.hull.Dd,min(self.hull.Bd,self.hull.halfBeam_MidBody(0.1*self.hull.Dd)]
    
        self.Lb = [0,0,0,0]
        self.Ub = [self.hull.LOA,self.hull.LOA,self.hull.Dd,self.hull.Bd]
        
        
        
        # Set variable types (bin is binary or boolean, real is floating point, and int is integer) 
        self.mask = ["real"]  * (4)
        
        super().__init__(n_var = 4, n_obj=1, xl=self.Lb, xu=self.Ub, n_constr = 5)
        # Set up problem for pymoo
        
  
                           
    def _evaluate(self, inputs, out, *args, **kwargs):
                
        
        out["F"] = -BoxVol(inputs)
        out["G"] = np.concatenate((BoxCons(inputs,self.hull),[PC_BoxCon(self.PC,inputs,self.hull.Dd)]))




#set up optimization code a call function:

def Run_BoxOpt(x):
    
    #returns [Fwd start of box, length of box, height, width, volume, final point cloud check if points exist outside the box (0 if good, 1 if fail)]
    #The first five terms are normalized by LOA(for lengths) or LOA^3(for volume)
    problem = BoxOpt(x)

    algorithm = NelderMead()

    res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)
    A = np.zeros((6,))
    
    A[0:4] = res.X/x[0]
    A[1] = A[1]-A[0]
    
    A[4] = -res.F/(x[0]**3.0)
    
    hull = HP(x)
    
    #PC = hull.gen_MeshGridPointCloud(NUM_WL = 201, PointsPerLOA = 501, Z = [], X = [], bit_GridOrList = 0)
    
    #A[5] = PC_BoxCon(PC, res.X,hull.Dd)
    
    return A
    

