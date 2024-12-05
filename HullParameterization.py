 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:50:30 2022

@author: shannon

Code Written by Noah Bagazinski
=============================================================================================

The Target of this Script is to Define the functions and parameters to define a ship hull. 

This Code is a continuation of Noah Bagazinski's work in ship hull genneration using a 
parametric design scheme.  

=============================================================================================
The hull parameterization is defined in five chunks:
    
1) General Hull Form. 
    
    This includes metrics such as the beam  at the deck, the 
    hull taper, and the stern taper.
    
2) The parallel mid body cross section. 
    
    This includes parameters for the flare, chine radius, deadrise, and keel radius
    
3) The Bow Form. 
    
    This includes functions that define the bow rise (rake), the profile drift angle with respect to depth,
    the profile of rocker at the front of the ship, and the profile of the location where the full breadth 
    is achieved for a given depth.
    
4) The Stern Form
    
   This includes functions that define the stern profile, the convergence angle (slope of the ship at the transom), the transom cross section, the profile
   of the rocker at the stern of the ship, the profile where the full beadth of the ship is achieved, and the curvature of the hull along the stern.
    
5) Bulb Forms
    
    Bulb forms slightly modified from the parameterization defined by:
        Chrismianto, D. and Kim, D.J., 2014. 'Parametric bulbous bow design using the cubic 
        Bezier curve and curve-plane intersection method for the minimization of ship resistance 
        in CFD'. Journal of Marine Science and Technology, 19(4), pp.479-492.
        
 functions include generation of NURBS curves to generate the meshes of the bulbous bow and stern 
    
"""


#import all the goodies:
import numpy as np
# scipy.optimize import fsolve
from matplotlib import pyplot as plt

from stl import mesh



class Hull_Parameterization:
  
    #Define parameters of targethull
    def __init__(self, inputs):
        '''
        inputs is a numpy vector that represents the parameterization of a hull.
        the instantiation function generates all of the constants and factors used 
        in defining the parameterization of the hull.
        Most of the inputs are scaled to LOA in some form 
        '''
        
        self.LOA = inputs[0]
        self.Lb = inputs[1] *self.LOA
        self.Ls = inputs[2] *self.LOA
        self.Bd = inputs[3]/2.0 *self.LOA#half breadth
        self.Dd = inputs[4] *self.LOA
        self.Bs = inputs[5] *self.Bd    #half breadth, fraction of Bd
        self.WL = inputs[6] *self.Dd
        self.Bc = inputs[7]/2.0 *self.LOA #half breadth
        self.Beta = inputs[8]
        self.Rc = inputs[9]*self.Bc
        self.Rk = inputs[10]*self.Dd
        self.BOW = np.zeros((3,))
        self.BOW[0] = inputs[11]*0.5*self.Lb/self.Dd**2.0
        self.BOW[1] = inputs[12]*0.5*self.Lb/self.Dd
        self.BK = np.zeros((2,))
        self.BK[1] = inputs[13] *self.Dd #BK_z is an input - BK_x is solved for
        self.Kappa_BOW = inputs[14]
        self.DELTA_BOW = np.zeros((3,))
        self.DELTA_BOW[0] = inputs[15]*0.5*self.Lb/self.Dd**2.0
        self.DELTA_BOW[1] = inputs[16]*0.5*self.Lb/self.Dd
        self.DRIFT = np.zeros((3,))
        self.DRIFT[0] = inputs[17]*60.0/self.Dd**2.0
        self.DRIFT[1] = inputs[18]*60.0/self.Dd
        self.DRIFT[2] = inputs[19]
        self.bit_EP_S = inputs[20]
        self.bit_EP_T = inputs[21]
        self.TRANS = np.zeros((2,))
        self.TRANS[0] = inputs[22]
        self.SK = np.zeros((2,))
        self.SK[1] = inputs[23]
        self.Kappa_STERN = inputs[24]
        self.DELTA_STERN = np.zeros((3,))
        self.DELTA_STERN[0] = inputs[25]*0.5*self.Ls/self.Dd**2.0
        self.DELTA_STERN[1] = inputs[26]*0.5*self.Ls/self.Dd
        #self.RY_STERN = np.array(inputs[25:27])
        #self.RX_STERN = np.array(inputs[27:29])
        self.Beta_trans = inputs[27]
        self.Bc_trans = inputs[28]/2.0 *self.LOA # half breadth
        self.Rc_trans = inputs[29]*self.Bc_trans
        self.Rk_trans = inputs[30]*self.Dd*(1-self.SK[1])
        #self.CONVERGE = np.array(inputs[33:36])
        self.bit_BB = inputs[31]
        self.bit_SB = inputs[32]
        self.Lbb = inputs[33]
        self.Hbb = inputs[34]
        self.Bbb = inputs[35]
        self.Lbbm = inputs[36]
        self.Rbb = inputs[37]
        self.Kappa_SB = inputs[38]
        self.Lsb = inputs[39]
        self.HSBOA = inputs[40]
        self.Hsb = inputs[41]
        self.Bsb = inputs[42]
        self.Lsbm = inputs[43]
        self.Rsb = inputs[44]
                       
        #Generate and Check the Forms of the Overall Hull
        
        self.GenGeneralHullform()
        #C1 = print(self.GenralHullformConstraints())
        
        self.GenCrossSection()
       # C2 = print(self.CrossSectionConstraints())
        
        self.GenBowForm()
        #C3 = print(self.BowformConstraints())
        
        self.GenSternForm()
        #C4 = print(self.SternFormConstraints())
        
        self.GenBulbForms()
        #C5 = print(self.BulbFormConstraints())

        
        
    '''
    =======================================================================
                        Section 1: General Hull Form
    =======================================================================
    
    The General hull form is characterized by 5 characteristics:
        
        0) LOA -> length overall of the vessel in [m] or = 1
        1) Lb  -> length of the bow taper in [m] or fraction of LOA
        2) Ls  -> length of the stern taper in [m] or fraction of LOA
        3) Bd  -> Beam at the top deck of the vessel in [m] or fraction of LOA
        4) Dd  -> Depth of the vessel at the deck in [m] or fraction of LOA
        5) Bs  -> Beam at the stern in [m] or fraction of LOA
        6) WL  -> Waterline depts in [m] or fraction of LOA
        
    Constraints / NOTES to ensure realistic sizing/ shape of a hull: 
        0) The length of the parallel mid body is equal to LOA-Lb-Ls = Lm
        1) Lb + Ls <= LOA
        2) Bd is not necessarily the maximum beam of the vessel.  It is only the breadth of the 
            main deck. BOA is calculated in the Section 2: Cross Section 
        3) 0 <= Bs <= BOA
        4) Lb is used to define the limits of the bow taper from the forwardmost point on the 
            bow rake to the point where the parallel mid-body starts. The profile of the ship 
            at different waterlines is dependent of the other parameters defined later in the 
            parameterization.
        5) Ls is used to define the limits of the stern taper from the aftmost point on the 
            stern rake to the point where the parallel mid-body ends. The profile of the ship 
            at different waterlines is dependent of the other parameters defined later in the 
            parameterization. 
        6) WL < Dd
        7) All variables are positive or 0
    '''
    def GenGeneralHullform(self):
        '''
        This funciton computes the other form factors of the general hullform
        that can be calculate from the inputs
        '''
        self.Lm = self.LOA - self.Ls - self.Lb    
    
    def GenralHullformConstraints(self):
        '''
        This function checks that constraints are satisfied for the hullfrom.
        If no constraint violations are found, 
        '''
        C = np.array([-self.LOA + self.Ls+self.Lb,
                      self.WL - self.Dd])
        return C
    '''
    =======================================================================
                        Section 2: Cross Section
    =======================================================================
    
    The Cross Section is defined by the following inputs:
        0) Bd   -> The Beam at the Deck in [m] or fraction of LOA
        1) Dd   -> The Depth of the Deck in [m] or fraction of LOA
        2) Bc   -> The Beam at the Chine  (intersection) in [m] or fraction of LOA
        3) Dc   -> The Depth of the Chine (intersection) in [m] or fraction of LOA
        4) Beta -> The deadrise angle in degrees
        5) Rc   -> The Chine Radius in [m] or fraction of LOA
        6) Rk   -> The keel Radius in [m] or fraction of LOA
    
    Constraints/ NOTES to ensure realistic sizing/ shape of a hull: 
        0) 0 <= Dc < Dd
        1) Rc and Rk are agebraically limited to ensure that the radius can exist with the 
            given Bd,Dd,BcdC, and Beta values. 
    
    '''
    def GenCrossSection(self):
        '''
        This function calculates the constants and other form factors that will allow future
        analysis of the cross section.

        '''
        
        #(y,z) pair for center of keel radius
        self.Rk_Center = np.array([-self.Rk*(0.5 - 0.5*np.sign(self.Rk)), 
                                    self.Rk*(0.5 + 0.5*np.sign(self.Rk))])
        #(y,z) pair for intersection of keel radius and LG line at the transom
        self.Rk_LG_int = np.array([self.Rk_Center[0] + self.Rk*np.sin(np.pi*self.Beta/180.0),
                                        self.Rk_Center[1] - self.Rk*np.cos(np.pi*self.Beta/180.0)])
       
        
        #solve for the lower gunwhale line: A*z + B*y + C = 0
        A = np.array([[1.0, 1.0, 1.0],
                      [self.Rk_LG_int[1], self.Rk_LG_int[0], 1.0],
                      [-(self.Rk_LG_int[0]-self.Rk_Center[0]), (self.Rk_LG_int[1]-self.Rk_Center[1]), 0.0]])
        b = np.array([1.0, 0.0, 0.0])

        self.LG = np.linalg.solve(A,b)
        
        del A, b     
        
        self.Dc = -(self.LG[1]*self.Bc + self.LG[2])/self.LG[0]
     
        # Upper Gunwhale Line: A*z + B*y + C = 0, where UG = [A,B,C]
        A = np.array([[self.Dc, self.Bc, 1.0],
                      [self.Dd, self.Bd, 1.0],
                      [1.0, 1.0, 1.0]])
        
        b = np.array([0.0,0.0,1.0])
        
        self.UG = np.linalg.solve(A,b)
        
        del A, b
        
        # Calculate terms for the half beam of the cross section of the transom:
        self.Rc_Center = np.zeros((2,)) #(y,z) pair for center of chine radius at the transom
        self.Rc_UG_int = np.zeros((2,)) #(y,z) pair for intersection of chine radius and UG line at the transom
        self.Rc_LG_int = np.zeros((2,)) #(y,z) pair for intersection of chine radius and LG line at the transom
        
        #make math more readable to solve the chine
        A1 = self.UG[0]
        B1 = self.UG[1]
        theta = np.arctan2(-B1,A1)
        
        
        if theta < 0.0:
            theta = theta + np.pi
 
        beta = self.Beta*np.pi/180.0
        A2 = self.LG[0]
        B2 = self.LG[1]
        
        
        A = np.array([[B1, A1, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, B2, A2, 0.0, 0.0],
                      [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                      [0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                      [0.0, 0.0, 0.0, -1.0, 0.0, 1.0]])
      
        b = np.array([-self.UG[2],
                      -self.LG[2],
                      self.Rc*np.sin(theta),
                      self.Rc*np.cos(theta),
                      self.Rc*np.sin(beta),
                      self.Rc*np.cos(beta)])
        
        C = np.linalg.solve(A,b)
        
        self.Rc_UG_int = C[0:2]
        self.Rc_LG_int = C[2:4]
        self.Rc_Center = C[4:6]
   
        
    def CrossSectionConstraints(self):
        
        C = [-self.Rc_UG_int[1] + self.Dc,
             -self.Rc,
             -self.Bc,
             -self.Dc,
             self.Rc_LG_int[0] - self.Bc,
             self.Rk_LG_int[0] - self.Rc_LG_int[0],
             0.00000001 -np.abs(self.Rk)]
        return C
    
    def halfBeam_MidBody(self, z):
        # This funtion calculates the half beam of the cross section at a given height, z
        # If 0 > z or Dd < z, then the function returns -1 as an error
        
        if z < 0.0 or z > self.Dd:
            return -1
        elif z >= 0.0 and z < self.Rk_LG_int[1]:
            return np.sign(self.Rk)*np.sqrt((self.Rk**2) - (z-self.Rk_Center[1])**2) + self.Rk_Center[0]
        elif z >= self.Rk_LG_int[1] and z < self.Rc_LG_int[1]:
            return -(self.LG[0] * z + self.LG[2])/self.LG[1]
        elif z >= self.Rc_LG_int[1] and z < self.Rc_UG_int[1]:
            return np.sqrt((self.Rc**2) - (z-self.Rc_Center[1])**2) + self.Rc_Center[0]
        else:
            return -(self.UG[0] * z + self.UG[2])/self.UG[1]


    def plot_MidBody_CrossSection(self):
        
        # Plot intersection points in blue
        # Plot chine pt in green
        # Plot Center of Rc and Rk in red
        # half Beam(z) in black
        
        z = np.linspace(0.0, self.Dd, num = 200)
        y = np.zeros((200,))
        for i in range(0,len(z)):
            y[i] = self.halfBeam_MidBody(z[i])
        
        
        fig2, ax2 = plt.subplots()
        ax2.axis('equal')
        #plt.axis([0,10,0,10])
        
        ax2.plot([self.Bd, self.Rc_UG_int[0], self.Rc_LG_int[0], self.Rk_LG_int[0], 0.0], 
                 [self.Dd, self.Rc_UG_int[1], self.Rc_LG_int[1], self.Rk_LG_int[1], 0.0], 'o', color = 'blue')
        ax2.plot([self.Rc_Center[0], self.Rk_Center[0]], [self.Rc_Center[1], self.Rk_Center[1]],'o' ,color = 'red')  
        ax2.plot([self.Bc], [self.Dc],'o' ,color = 'green')  
        ax2.plot(y,z,'-', color = 'black', linewidth = 0.75)
        
    
    
    
    '''
    =======================================================================
                        Section 3: Bow Form
    =======================================================================
    
    The Bow Form is defined by the following inputs:
        0) Dd   -> The Depth of the Deck in [m] or fraction of LOA
        1) Lb   -> The length of the bow taper in [m] or fraction of LOA
        2) Abow  -> The z^2 term for Bow(z) that defines the profile of the bowrise
        3) Bbow  -> The z term for Bow(z) that defines the profile of the bowrise
        4) BK_z -> The Z Point of the intersection of the Bow rise and keel rise as percentage of Dd
        5) Kappa_bow-> The X position where the Keel rise begins. percentage of Lb
        6) Adel -> z^2 term for delta(z), the x position where the max Beam is achieved for a given height
        7) Bdel -> z term for delta(z), the x position where the max Beam is achieved for a given height
        8) Adrft-> z^2 term for drift(z), the drift angle along the bowrise and keel rise
        9) Bdrft-> z term for drift(z), the drift angle along the bowrise and keel rise
        10) Cdrft-> const term for drift(z), the drift angle along the bowrise and keel rise
    
    These Parameters solve for 4 functions:
        0) Bow(z)   -> gives the X position of the bow rise in the form Az^2 + Bz + C
        1) Keel_BOW(x)  -> gives the z height of the keel rise with respect to X in the form A*(X-Kappa_BOW*Lb)^2
        2) Delta_BOW(z) -> gives the x position between 0 and Lb where the full breadth is achieved for a given z: A(z/Dd)^2 + B(z/Dd) + C = x/Lb
        3) Drift(z) -> gives the drift angle of the bow for a given z: Az^2 + Bz + C
    
    These four functions define the following curve for each z:
        halfBeam_Bow(x) = Y(x) = A*x^3 + Bx^2 + Cx + D for all z between 0 and Dd
        Since we know two points and the derivatives of  those two points
    
    Constraints/ NOTES to ensure realistic sizing/ shape of a hull: 
        0) Kappa_BOW*Lb < delta(z=0)
        1) 0 < drift(z) < 90 for 0 <= z <= Dd (only need to check at z = 0, Dd, and -B/(2*A) if within range of z )
        2) 0 <= BK_x < Kappa_BOW*Lb
        3) 0 <= BK_z < Dd
        4) delta(z) > Bow(z) and Keel(z) for 0 <= z <= Dd  -> check z = 0,Dd,BK, Vert (Bow) and Vert (Delta)
    
    '''
    
    def GenBowForm(self):
        '''
        This funciton computes the other form factors of the Bowform
        that can be calculated from the inputs
        '''
            
        if self.BOW[0] == 0:
            Zv = -1.0
        else:
           Zv = -self.BOW[1]/(2*self.BOW[0]) #Find Z of vertex of bowrise(z)
        
        C = np.array([self.BOW[0]*self.Dd**2.0 + self.BOW[1]*self.Dd,   #Bow rise protrusion at Deck
                      self.BOW[0]*self.BK[1]**2.0 + self.BOW[1]*self.BK[1], #Bow rise protrusion at Bow-keel  intersection
                      self.BOW[0]*Zv**2.0 + self.BOW[1]*Zv]) #Bowrise protrusio at vertex of bow rise eqn
                      
        if (Zv >= self.BK[1]*self.Dd and Zv <= self.Dd):
            self.BOW[2] = -np.amin(C) # If the vertex is between the BK intersect and the Deck, then it is included in the min search
        else:
            self.BOW[2] = -np.amin(C[0:2])
            
        
        # X Position of BK intersect
        self.BK[0] = self.bowrise(self.BK[1])
        
        
        # Calculate the Keelrise equation: it is of the form X = sqrt(Z/A) + Kappa_Bow*Lb or Z = A(X-K*Lb)**2, where self.Keel = A
        self.KEEL_BOW = self.BK[1]/((self.BK[0]-self.Kappa_BOW*self.Lb)**2.0)
        
        
        #Calculate the C for the Delta equation, where C is the constant such that max(Delta(z)) = 0 between 0 and Dd
        if self.DELTA_BOW[0] == 0:
            Zv = -1.0
        else:
            Zv = -self.DELTA_BOW[1]/(2*self.DELTA_BOW[0]) #Find Z of vertex of Delta(z)
        
        C = np.array([self.DELTA_BOW[0]*self.Dd**2.0 + self.DELTA_BOW[1]*self.Dd,   #BDelta at Deck
                      0.0, #As is, Delta(0) = 0
                      self.DELTA_BOW[0]*Zv**2.0 + self.DELTA_BOW[1]*Zv]) #Bowrise protrusion at vertex of bow rise eqn
                      
        if (Zv >= 0.0 and Zv <= self.Dd):
            self.DELTA_BOW[2] = -np.amax(C) # If the vertex is between z = 0  and the Deck, then it is included in the search
        else:
            self.DELTA_BOW[2] = -np.amax(C[0:2])      
    
    #The following funcitons return the 
        
    def bowrise(self, z):
        #returns the x position of the bowrise for a given z for BK_z <= z <= Dd
        return self.BOW[0]*z**2.0 + self.BOW[1]*z + self.BOW[2]
    
    def keelrise_bow(self, z):
        #returns the x position of the keelrise  at the bow for a given z for 0 <= z <= Bk_z
        return -np.sqrt(z/self.KEEL_BOW) + self.Kappa_BOW*self.Lb
    
    def delta_bow(self, z):
        #returns the x position where the full cross section width is achieved for a given z for 0 <= z <= Dd
        return self.Lb + self.DELTA_BOW[0]*z**2.0 + self.DELTA_BOW[1]*z + self.DELTA_BOW[2]
    
    
    def drift(self, z):
        #returns the drift angle in radians
        return np.pi*(self.DRIFT[0]*z**2.0 + self.DRIFT[1]*z + self.DRIFT[2])/180.0
        
    def solve_waterline_bow(self,z):
        #this function solves for a cubic function: y(half beam) = Ax^3 + Bx^2 + CX + D for the half beam of the profile between the bow/keel rise and delta for a given z for 0 <= z <= Dd

        X1 = self.bow_profile(z)
        
        X2 = self.delta_bow(z)
        
        Y2 = self.halfBeam_MidBody(z)
        
        A = np.array([[X1**3.0, X1**2.0, X1, 1.0],
                      [3.0*X1**2.0, 2*X1, 1.0, 0.0],
                      [X2**3.0, X2**2.0, X2, 1.0],
                      [3.0*X2**2.0, 2.0*X2, 1.0, 0.0]])
            
        b = np.array([0.0,
                      np.tan(self.drift(z)),
                      Y2,
                      0.0])
        return np.linalg.solve(A,b)
    
    def bow_profile(self, z):
        # This assumes that z >= 0 and z <= Dd
        
        if z <= self.BK[1]:
            X1 = self.keelrise_bow(z)
        else:
            X1 = self.bowrise(z)
        return X1
    
    def halfBeam_Bow(self, x, PROF):
        #returns the halfbeam along the bow taper between the bow/keel rise and delta(z), PROF is the output of solve)waterline_bow(z)
        #x is a vector
        y = np.zeros((len(x),))
        for i in range(0,len(x)):
            y[i] = PROF[0]*x[i]**3.0 + PROF[1]*x[i]**2.0 + PROF[2]*x[i] + PROF[3]
        return y
    
    def bow_dydx(self, x, PROF):
        #returns slope dydx of the bow taper at a height z that is defined by PROF
        #x is a vecotr and function returns the vector of dydx
        
        dydx = np.zeros((len(x),))
        
        for i in range(0,len(x)):
            dydx[i] = 3.0*PROF[0]*x[i]**2.0 + 2.0*PROF[1]*x[i] + PROF[2]
        
        return dydx
    
    
    def gen_waterline_bow(self, z, NUM_POINTS = 100, X = [0,1], bit_spaceOrGrid = 1):
        '''
        This fuction generates a set of points [[X1,Y1] .... [X2,Y2]] that detail the curvature of the bow taper for a given z, for 0 <= z <= Dd
        
        it can either be created as with a number of set of points, or an even spacing based on the global x spacing (better for station plotting) 

        BOOL_PTS_OR_SPACE controls whether a set number of points will produce the waterline (1) or the spacing vector will(0)
        
        '''
       
        x1 = self.bow_profile(z)
        
        x2 = self.delta_bow(z)
        
        prof = self.solve_waterline_bow(z)
        
            #Set x based on spacing or grid
        if bit_spaceOrGrid:
            
            x = np.linspace(x1,x2,NUM_POINTS)
            XY = np.zeros((len(x),2))
                        
        else:
            x = [i for i in X if (i > x1 and i <= x2)]
            
            
            x = np.concatenate(([x1],x))
            XY = np.zeros((len(x),2))
            
            
        XY[0,:] = [x1, 0.0]
        
        y = self.halfBeam_Bow(x[1:], prof)

        XY[1:] = np.transpose([x[1:],y])
            
        return XY
        
    def BowformConstraints(self):
        #This fuction returns booleans if the bow constraints are satisfied as detailed above:
            
        #Check that the vertex (Zv) of the drift angle equation satisfies the constraint of the drift angle if
        # if it lies within the bounds of 0 and Dd. If drift(z) is a line, or the vertex is outside the bounds,
        # Then True is returned 
        if self.DRIFT[0] == 0.0:
            Zv = -1.0
        else:
            Zv = -self.DRIFT[1]/(2.0*self.DRIFT[0])
        
        
        if Zv >= 0.0 and Zv <= self.Dd:
            vert_drift = [self.drift(Zv) - np.pi/2.0,
                          -self.drift(Zv)]
        else:
            vert_drift = [-1,-1]
            
            
        #Check that Delta_Bow(z) is always greater than the leading edge of the ship (keelrise(z) and bow(z))
        #Check at z = 0, vertex of delta(z), vertex of bow(z), BKz, Dd
        if self.DELTA_BOW[0] == 0.0:
            Zv = -1.0
        else: 
            Zv = -self.DELTA_BOW[1]/ (2.0*self.DELTA_BOW[0])
        
        if Zv >=0.0 and Zv <= self.Dd:
            vert_delta_bow = (-self.delta_bow(Zv) + self.bow_profile(Zv))
        else:
            vert_delta_bow = -1
        
        
        if self.BOW[0] == 0.0:
            Zv = -1.0
        else: 
            Zv = -self.BOW[1]/ (2.0*self.BOW[0])
        
        if Zv >=0.0 and Zv <= self.Dd:
            vert_bow = (-self.delta_bow(Zv) + self.bow_profile(Zv))
        else:
            vert_bow = -1
        
        
        
        
        C = [self.Kappa_BOW*self.Lb - self.delta_bow(0.0),
            self.drift(0.0) - np.pi/2.0,
            -self.drift(0.0) ,
            self.drift(self.Dd) - np.pi/2.0,
            -self.drift(self.Dd),
            vert_drift[0],
            vert_drift[1],
            -self.BK[0],
            self.BK[0] - self.Kappa_BOW*self.Lb,
            -self.BK[1],
            self.BK[1] - self.Dd,
            -self.delta_bow(self.Dd) + self.bow_profile(self.Dd),
            -self.delta_bow(self.BK[1]) + self.BK[0],
            vert_delta_bow,
            vert_bow]
        return C
    
    
    '''
    =======================================================================
                        Section 4: Stern Form
    =======================================================================
        The Stern Form is defined by the following inputs:
        0) bit_EP_S -> defines whether the stern will be elliptical (1) or parabolic (0) below the SK intersect
        1) bit_EP_T -. Defines whether the stern will be elliptical (1) or parabolic (0) abover the SK intersect
        2) Bs   -> The width of the stern at the deck of the ship in [m] or fraction of LOA
        3) Ls   -> The length of the stern taper in [m] or fraction of LOA
        4) A_trans -> The A term that defines the transom slope X = Az + B
        5) SKz -> The Z Point of the intersection of the Stern rise and transom as percentage of Dd
        6) Kappa_STERN -> The X position where the Stern rise begins aft of the end of the parallel midbody as a fraction of Ls
        7) Adel -> z^2 term for delta_stern(z), the x position where the max Beam is achieved for a given height,
        8) Bdel -> z term for delta_stern(z), the x position where the max Beam is achieved for a given height
        9) Bc_trans -> The beam of the chine point at the transom in [m] or fraction of LOA
        10) Dc_trans -> The depth of the chine point at the transom in [m] or fraction of LOA
        11) Rc_trans -> The Chine radius of the chine at the transom in [m] or fraction of LOA
        12) Rk_trans -> the keel radius of the chine at the transom in [m] or fraction of LOA
    
    
    
        REMOVE THESE FOR NOW 
        7) A_Ry-> z term for  Ry(z), the y-raduis of the ellipse at the stern of the ship
        8) B_Ry-> const for Ry(z), the y-raduis of the ellipse at the stern of the ship
        9) A_Rx-> z term for  Rx(z), the x-raduis of the ellipse at the stern of the ship
        10) B_Rx-> const for Rx(z), the x-raduis of the ellipse at the stern of the ship

        15) AconvT -> the z^2 term for Converge Angle(z) the tangent angle of the gunwhale at the transom
        16) BconvT -> the z term for Converge Angle(z) the tangent angle of the gunwhale at the transom
        17) CconvT -> the const term for Converge Angle(z) the tangent angle of the gunwhale at the transom
    
    
    These Parameters solve for 7 functions:
        0) Transom(z)   -> gives the X position of the transom in the form  Az + B
        1) Sternrise(x)  -> gives the z height of the stern rise with respect to X in the form A*(X-Kappa*Ls)^2
        2) Delta_Stern(z) -> gives the x position between LOA-Ls and LOA where the full breadth is achieved for a given z: A(z)^2 + B(z) + C = X
        3) halfBeam_transom(z) -> gives the halfbeam of the transom for z between SKz and Dd
    
        REMOVE THESE FOR BIW
        3) Converge(z) -> gives the convergence tangent angle of the gunwhale at the transom for a given z: Az^2 + Bz + C
        4) Ry(z) -> gives the y radius of the stern ellipse in the form Ry = Az + B
        5) Rx(z) -> gives the x radius of the stern ellipse in the form Rx = Az + B
        
    
    These four functions define the following curve for each z:
        halfBeam_Stern(x) = Y(x) = Parabola + Ellipse for all z between 0 and Dd
    
    Constraints/ NOTES to ensure realistic sizing/ shape of a hull: 
        0) Lb+Lm + Kappa_Stern*Ls > delta_Stern(z=0)
        1) 0 < converge(z) < 90 for 0 <= z <= Dd (only need to check at z = 0, Dd, and -B/(2*A) if within range of z )
        2) 0 <= SK_x > Lb+Lm+ Kappa*Ls
        3) 0 <= SK_z < Dd
        4) delta(z) < Transom(z) and Sternrise(z) for 0 <= z <= Dd 
    
    
    
    '''
    def GenSternForm(self):
        
        # Recalculate SK to be a value instead of a percentage
        self.SK[1] = self.SK[1]*self.Dd
        
        # Solve for the B value such that max(Transom(z)) = LOA
        if self.TRANS[0] >= 0.0:
            self.TRANS[1] = self.LOA - self.TRANS[0]*self.Dd
        else:
            self.TRANS[1] = self.LOA - self.TRANS[0]*self.SK[1]
        
        #calculate the x value for the SK intersect
        self.SK[0] = self.transom(self.SK[1])
        
        # find the constant term in the sternrise equation: z = A(x-Lb+Lm+Ls*Kappa_stern)^2
        self.STERNRISE = self.SK[1]/(self.SK[0] - (self.Lb + self.Lm + self.Ls*self.Kappa_STERN))**2.0
        
        #Calculate the C for the Delta_stern equation, where C is the constant such that min(Delta_stern(z)) = 0 for z between 0 and Dd
        if self.DELTA_STERN[0] == 0:
            Zv = -1.0
        else:
            Zv = -self.DELTA_STERN[1]/(2*self.DELTA_STERN[0]) #Find Z of vertex of Delta(z)
        
        C = np.array([self.DELTA_STERN[0]*self.Dd**2.0 + self.DELTA_STERN[1]*self.Dd,   #Stern Delta at Deck
                      0.0, #As is, Delta_Stern(0) = 0
                      self.DELTA_STERN[0]*Zv**2.0 + self.DELTA_STERN[1]*Zv]) #vertex of Delta_STERN equation
                      
        if (Zv >= 0.0 and Zv <= self.Dd):
            self.DELTA_STERN[2] = -np.amin(C) # If the vertex is between z = 0  and the Deck, then it is included in the search
        else:
            self.DELTA_STERN[2] = -np.amin(C[0:2])
        
        
        #(y,z) pair for center of keel radius
        self.Rk_Center_trans = np.array([-self.Rk_trans*(0.5 - 0.5*np.sign(self.Rk_trans)), 
                                         self.SK[1] + self.Rk_trans*(0.5 + 0.5*np.sign(self.Rk_trans))])
        #(y,z) pair for intersection of keel radius and LG line at the transom
        self.Rk_LG_int_trans = np.array([self.Rk_Center_trans[0] + self.Rk_trans*np.sin(np.pi*self.Beta_trans/180.0),
                                        self.Rk_Center_trans[1] - self.Rk_trans*np.cos(np.pi*self.Beta_trans/180.0)])
       
        
        #solve for the lower gunwhale line: A*z + B*y + C = 0
        A = np.array([[1.0, 1.0, 1.0],
                      [self.Rk_LG_int_trans[1], self.Rk_LG_int_trans[0], 1.0],
                      [-(self.Rk_LG_int_trans[0]-self.Rk_Center_trans[0]), (self.Rk_LG_int_trans[1]-self.Rk_Center_trans[1]), 0.0]])
        b = np.array([1.0, 0.0, 0.0])

        self.LG_trans = np.linalg.solve(A,b)
        
        del A, b     
        
        self.Dc_trans = -(self.LG_trans[1]*self.Bc_trans + self.LG_trans[2])/self.LG_trans[0]
     
        # Upper Gunwhale Line: A*z + B*y + C = 0, where UG = [A,B,C]
        A = np.array([[self.Dc_trans, self.Bc_trans , 1.0],
                      [self.Dd, self.Bs, 1.0],
                      [1.0, 1.0, 1.0]])
        
        b = np.array([0.0,0.0,1.0])
        
        self.UG_trans = np.linalg.solve(A,b)
        
        del A, b
        
        # Calculate terms for the half beam of the cross section of the transom:
        self.Rc_Center_trans = np.zeros((2,)) #(y,z) pair for center of chine radius at the transom
        self.Rc_UG_int_trans = np.zeros((2,)) #(y,z) pair for intersection of chine radius and UG line at the transom
        self.Rc_LG_int_trans = np.zeros((2,)) #(y,z) pair for intersection of chine radius and LG line at the transom
        
        #make math more readable to solve the chine
        A1 = self.UG_trans[0]
        B1 = self.UG_trans[1]
        theta = np.arctan2(-B1,A1)
        
        if theta < 0.0:
            theta = theta + np.pi
        
        beta = self.Beta_trans*np.pi/180.0
        A2 = self.LG_trans[0]
        B2 = self.LG_trans[1]
        
        
        A = np.array([[B1, A1, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, B2, A2, 0.0, 0.0],
                      [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                      [0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                      [0.0, 0.0, 0.0, -1.0, 0.0, 1.0]])
      
        b = np.array([-self.UG_trans[2],
                      -self.LG_trans[2],
                      self.Rc_trans*np.sin(theta),
                      self.Rc_trans*np.cos(theta),
                      self.Rc_trans*np.sin(beta),
                      self.Rc_trans*np.cos(beta)])
        
        C = np.linalg.solve(A,b)
        
        self.Rc_UG_int_trans = C[0:2]
        self.Rc_LG_int_trans = C[2:4]
        self.Rc_Center_trans = C[4:6]
        
        
        
    
    def transom(self, z):
        #returns the x position of the transom for a given z fr SK_z <= z <= Dd
        return self.TRANS[0]*z + self.TRANS[1]    
    
    def sternrise(self, z):
        #returns the x position of the sternrise for a given z for 0 <= z <= SK_z
        return np.sqrt(z/self.STERNRISE) + self.Lb + self.Lm + self.Ls*self.Kappa_STERN
    
    def stern_profile(self, z):
        # shows the profile of the stern without the bulbous stern:
            
        if self.bit_SB and z <= self.WL*self.HSBOA:
            return self.SB_Prof[0] # If there is a bulbous stern, we want to form the profile to lead into the SB.
        
        else:
            if z <= self.SK[1]:
                return self.sternrise(z)
            else:
                return self.transom(z)
    
    def delta_stern(self, z):
        #returns the starting position of the stern taper at a given heigt
        return self.Lb + self.Lm + self.DELTA_STERN[0]* z**2.0 + self.DELTA_STERN[1]*z + self.DELTA_STERN[2]
   

    def halfBeam_Transom(self, z):
        #Returns the x,y pair of the transom at a height z. This assumes that SK_z <= z <= Dd. otherwise y returns -1
        x = self.stern_profile(z)
        
        if z <= self.SK[1] and z > 0.0:
            y = 0.0
        elif z < 0.0 or z > self.Dd:
            y = -1.0
        
        elif z > self.SK[1] and z < self.Rk_LG_int_trans[1]:
            y =  np.sign(self.Rk_trans)*np.sqrt((self.Rk_trans**2) - (z-self.Rk_Center_trans[1])**2) + self.Rk_Center_trans[0]
        elif z >= self.Rk_LG_int_trans[1] and z < self.Rc_LG_int_trans[1]:
            y =  -(self.LG_trans[0] * z + self.LG_trans[2])/self.LG_trans[1]
        elif z >= self.Rc_LG_int_trans[1] and z < self.Rc_UG_int_trans[1]:
            y =  np.sqrt((self.Rc_trans**2) - (z-self.Rc_Center_trans[1])**2) + self.Rc_Center_trans[0]
        else:
            y =  -(self.UG_trans[0] * z + self.UG_trans[2])/self.UG_trans[1]
         
        return [x,y]
    
    def plot_Transom_CrossSection(self):
        
        # Plot intersection points in blue
        # Plot chine pt in green
        # Plot Center of Rc and Rk in red
        # half Beam(z) in black
        
        z = np.linspace(self.SK[1], self.Dd, num = 200)
        y = np.zeros((200,2))
        for i in range(0,len(z)):
            y[i] = self.halfBeam_Transom(z[i])
        
        
        fig1,ax1 = plt.subplots()
        ax1.axis('equal')
        #plt.axis([0,10,0,10])
        
        ax1.plot([self.Bs, self.Rc_UG_int_trans[0], self.Rc_LG_int_trans[0], self.Rk_LG_int_trans[0], 0.0], 
                 [self.Dd, self.Rc_UG_int_trans[1], self.Rc_LG_int_trans[1], self.Rk_LG_int_trans[1], self.SK[1]], 'o', color = 'blue')
        ax1.plot([self.Rc_Center_trans[0], self.Rk_Center_trans[0]], [self.Rc_Center_trans[1], self.Rk_Center_trans[1]],'o' ,color = 'red')  
        ax1.plot([self.Bc_trans], [self.Dc_trans],'o' ,color = 'green')  
        ax1.plot(y[:,1],z,'-', color = 'black', linewidth = 0.75)
    
    def halfBeam_Stern(self, x, PROF):
        #returns the halfbeam along the stern taper between delta(z) and stern_profile(z), PROF is the output of solve_waterline_stern(z)
        # x is a vector
        y = np.zeros((len(x),))
        
        if PROF[0]:
            for i in range(0,len(x)):
                y[i] = np.sqrt( np.abs(PROF[5]**2.0 * (1.0 - ((x[i] - PROF[6])/PROF[4])**2.0))) + PROF[7]  #ellipse
        else:
            for i in range(0,len(x)):
                
                y[i] = PROF[1]*x[i]**2.0 + PROF[2]*x[i] + PROF[3]  #parabola
     
        return y
    
    def stern_dydx(self, x, PROF):
        #returns slope dydx of the stern taper at a height z that is defined by PROF
        #x is a vecotr and function returns the vector of dydx
        
        dydx = np.zeros((len(x),))
        
        if PROF[0]:
            for i in range(0,len(x)):  
                
                dydx[i] = -PROF[5]*(x[i]-PROF[6])/(PROF[4]**2.0) * 1/np.sqrt(np.abs(1.0 - ((x[i] - PROF[6])/PROF[4])**2.0))
            
        else:
            for i in range(0,len(x)):
                
                dydx[i] = 2.0*PROF[1]*x[i] + PROF[2]
        
        return dydx
        
    
    
    def solve_waterline_stern(self, z):
        #returns PROF, a parabola [A,B,C], an ellipse [Rx, Ry, Cx, Cy] of the two curves such that they are tangent at the intersection
        # Compares bit_EP_S and bit_EP_T -> if  the curve is a parabola, the ellipse values in PROF are 0 and vice versa
        # PROF = [A,B,C, Rx, Ry, Cx, Cy]
        
        x1 = self.delta_stern(z)
        y1 = self.halfBeam_MidBody(z)
        
        [x2,y2] = self.halfBeam_Transom(z)
        
        PROF = np.zeros((8,))
        
        if z >= self.SK[1]:
            if self.bit_EP_T:
                #If the curve is at the transom and the curve is an ellipse
                Rx = x2-x1
                Ry = y1-y2
                Cx = x1
                Cy = y2
                PROF[0] = 1
                PROF[4:8] = np.array([Rx,Ry,Cx,Cy])
            else:
                #If the curve is at the transom and the curve is a parabola:
                A = np.array([[x1**2.0, x1, 1.0],
                              [x2**2.0, x2, 1.0],
                              [2.0*x1, 1.0, 0.0]])
                
                b = np.array([y1, y2, 0.0])
                
                C = np.linalg.solve(A,b)
                
                PROF[1:4] = C
        else:
            if self.bit_EP_S:
                #If the curve is below the transom and the curve is an ellipse
                Rx = x2-x1
                Ry = y1-y2
                Cx = x1
                Cy = y2
                PROF[0] = 1
                PROF[4:8] = np.array([Rx,Ry,Cx,Cy])
            else:
                #If the curve is below the transom and the curve is a parabola:
                A = np.array([[x1**2.0, x1, 1.0],
                              [x2**2.0, x2, 1.0],
                              [2.0*x1, 1.0, 0.0]])
                
                b = np.array([y1, y2, 0.0])
                
                C = np.linalg.solve(A,b)
                
                PROF[1:4] = C
                
        return PROF
        

    
    def gen_waterline_stern(self, z, NUM_POINTS = 100, X = [0,1], bit_spaceOrGrid = 1):
        '''
        This fuction generates a set of points [[X1,Y1] .... [X2,Y2]] that detail the curvature of the bow taper for a given z, for 0 <= z <= Dd
        
        it can either be created as with a number of set of points, or an even spacing based on the global x spacing (better for station plotting) 

        BOOL_PTS_OR_SPACE controls whether a set number of points will produce the waterline (1) or the spacing vector will(0)
        
        '''
        x1 = self.delta_stern(z)
        
        x2 = self.stern_profile(z)
        
        
        
        prof = self.solve_waterline_stern(z)
               
        
        
        if bit_spaceOrGrid:
            x = np.linspace(x1,x2,NUM_POINTS)
            XY = np.zeros((len(x),2))
            
        else:
            x = [i for i in X if (i >= x1 and i < x2)]
            x = np.concatenate((x,[x2]))
            XY = np.zeros((len(x),2))
          
        y = self.halfBeam_Stern(x[0:-1], prof)
        
   
        XY[0:-1] = np.transpose([x[0:-1],y])
              
        #set the last element in the array to be the transom point
        XY[-1] = self.halfBeam_Transom(z)
                
            
        return XY
      
    
    
    def SternformConstraints(self):
        # this is an incomplete list of geometric constrains for the hull form
        
        #Check that Delta_Bow(z) is always greater than the leading edge of the ship (keelrise(z) and bow(z))
        #Check at z = 0, vertex of delta(z), vertex of bow(z), BKz, Dd
        if self.DELTA_STERN[0] == 0.0:
            Zv = -1.0
        else: 
            Zv = -self.DELTA_STERN[1]/ (2.0*self.DELTA_STERN[0])
        
        if Zv >=0.0 and Zv <= self.Dd:
            vert_delta_stern = (self.delta_stern(Zv) - self.stern_profile(Zv))
        else:
            vert_delta_stern = -1
        
        
        
        C = [self.delta_stern(0.0) - (self.Lb + self.Lm + self.Ls*self.Kappa_STERN),
             self.delta_stern(self.SK[1]) - self.SK[0],
             vert_delta_stern,
             self.delta_stern(self.Dd) - self.stern_profile(self.Dd),
             (self.Lb + self.Lm + self.Ls*self.Kappa_STERN) - self.SK[0],
             self.Bc_trans - self.halfBeam_MidBody(self.Dc_trans),
             -self.Rc_UG_int_trans[1] + self.Dc_trans,
             -self.Rc_trans,
             -self.Bc_trans,
             -self.Dc_trans,
             self.Rc_LG_int_trans[0] - self.Bc_trans,
             self.Rk_LG_int_trans[0] - self.Rc_LG_int_trans[0]]
         
    
        return C
    
    
    
    
    '''
    =======================================================================
                        Section 5: Bulb Forms
    =======================================================================
    The Bulb Forms are defined by the following inputs:
        0)  bit_BB   -> Bit that defines whether there is a bublous bow (1) or not (0)
        1)  bit_SB   -> Bit that defines whether there is a bublous stern (1) or not (0)
        2)  Lbb      -> Length of the bulbous bow (BB) fwd the foward perpendicular as a fraction of LOA
        3)  Hbb      -> Height of widest part of the BB as a fraction of the WL
        4)  Bbb      -> max. Width of the BB at the FP as a fraction of Bd
        5)  Lbbm     -> A midpoint along the length of the BB as a fraction of Lbb -> represents the position of max beam of the BB
        6)  Rbb      -> radius that fillets the BB to the hull (ratio from 0 to 1) -> solved as a cubic function
        7)  Kappa_SB -> Position where the Stern Bulb diverges from the hull as a fraction of Ls
        8)  Lsb      -> Length of the stern bulb (SB) as a fraction of LOA
        9) HSBOA     -> Overall Height of the SB as a fraction of WL
        9)  Hsb      -> Height of widest part of the SB as a fraction of HSBOA
        10) Bsb      -> max. Width of the SB at the Kappa_SB as a fraction of Bd
        11) Lsbm     -> A midpoint along the length of the SB as a fraction of Lsb -> represents the position of max beam of the BB
        12) Rsb      -> radius that fillets the SB to the hull (ratio from 0 to 1) -> solved as a cubic function

    
    
    These Parameters solve 3 functions each for the Bulbous Bow and Bulbous Stern:
       0) Outline: Definition of upper and lower ellipse that define the profile of the bulb
       1) Profile of Max width : a parabola that is tangent to an ellipse at the  longitudinal mid point.
       2) Cross Section Generator: Solves for Rx and Ry of an upper and lower ellipse that solves for a cross section of the bulb
       3) 
    
    
    Constraints/ NOTES to ensure realistic sizing/ shape of a bulb: 
        0) Cross Section of bulbs at Starting position need to be encompassed by Half Beam Mid Body
        1) Rk >= 0
        2) 0.0 < (Hbb and Hsb) < 1.0
        3) 0.0 < (Bbb and Bsb) < 1.0
        4) 0.0 < (Lbb and Lsb) < TBD  (seems a bit outrageous) (but not infeasible technically)
        5) -1.0 < (Lbbm and Lsbm) < 1.0
        6) 0.0 < HSBOA < 1.0
    
    
    

    
    '''
    def GenBulbForms(self):
        '''
        This function generates Prof for the bulbous bow and bulbous stern.
        
        it is composted of the following elements in this order:
            0a) FP = forward perpendicular: the start point were the BB diverges from the hull. 
            0b) SBs = start of stern bulb: the starting point where the SB diverges from the hull
            1)  Rz_U = upper z radius for an ellipse that defines the top half of the bulb
            2)  Rz_L = lower z radius for an ellipse that defines the bottom half of the bulb
            3)  Ry -> y radius of an ellipse that defines the max width of the bulb
            4)  Rx -> x Radius of leading edge of bulb
            5)  Cx -> X center of aforementioned ellipse
            6)  Cy -> Y center of aforementioned ellipse

        
        '''

        self.BB_Prof = np.zeros((7,))
        self.SB_Prof = np.zeros((7,))
        
        if self.bit_BB:
            
            FP = self.bow_profile(self.WL)
                        
            
            self.BB_Prof[0] = FP
            self.BB_Prof[1] = (1.0 - self.Hbb)*self.WL
            self.BB_Prof[2] = self.Hbb*self.WL
            self.BB_Prof[3] = self.halfBeam_MidBody(self.BB_Prof[2]) *self.Bbb
            self.BB_Prof[4] = self.Lbb*self.LOA*(1.0-self.Lbbm)
            self.BB_Prof[5] = FP - self.LOA*self.Lbb*self.Lbbm
            self.BB_Prof[6] = 0.0
        
        
        if self.bit_SB:
            SBs = self.Kappa_SB*self.Ls + self.Lm + self.Lb
            
            self.SB_Prof[0] = SBs
            self.SB_Prof[1] = (1.0 - self.Hsb)*self.WL*self.HSBOA
            self.SB_Prof[2] = self.Hsb*self.WL*self.HSBOA
            self.SB_Prof[3] = self.halfBeam_MidBody(self.SB_Prof[2])*self.Bsb
            self.SB_Prof[4] = self.Lsb*self.LOA*(1.0-self.Lsbm)
            self.SB_Prof[5] = SBs + self.LOA*self.Lsb*self.Lsbm
            self.SB_Prof[6] = 0.0
    
    
    def BB_profile(self,z):
        #assumes z is between WL  and 0.0
        #returns x position of leading edge of SB
        if z >= self.BB_Prof[2]:
            return self.BB_Prof[5] - np.sqrt(np.abs(1.0 - ((z-self.Hbb*self.WL)/self.BB_Prof[1])**2.0))*self.BB_Prof[4]
        else:
            return self.BB_Prof[5] - np.sqrt(np.abs(1.0 - ((z-self.Hbb*self.WL)/self.BB_Prof[2])**2.0))*self.BB_Prof[4]
        
    def halfBeam_BB(self, z, x):
        #returns the half breadth of the BB at height z and position x. x is a vector
        #assumes z is between 0 and WL assumes x greater than BB_profile(z)
        
        if z >= self.BB_Prof[2]:
            Rz = self.BB_Prof[1]
        else:
            Rz = self.BB_Prof[2]
        
        Ry = np.sqrt(np.abs(1.0 - ((z - self.BB_Prof[2])/Rz)**2.0))*self.BB_Prof[3]
        
        Rx = self.BB_Prof[5] - self.BB_profile(z)
        
        y = np.zeros((len(x),))
        
        for i in range(0,len(x)):
            
            if x[i] >= self.BB_Prof[5]:
                y[i] = Ry
            else:
                y[i] = np.sqrt(np.abs(1.0 - ((x[i] - self.BB_Prof[5])/Rx)**2.0))*Ry
        
        return y
        
        
        
    def BB_dydx(self,z,x):
        #This function computes the slope dy/dx slope of the bulbous bow at height z and position x. This assumes x is within the bulbous bow x-range
        # x is a vector
        
        dydx = np.zeros((len(x),))
        
        if z >= self.BB_Prof[2]:
            Rz = self.BB_Prof[1]
        else:
            Rz = self.BB_Prof[2]
        
        Ry = np.sqrt(np.abs(1.0 - ((z - self.BB_Prof[2])/Rz)**2.0))*self.BB_Prof[3]
        Rx = self.BB_Prof[5] - self.BB_profile(z)
        
        for i in range(0,len(x)):
        
            if x[i] >= self.BB_Prof[5]:
                dydx[i] = 0.0
            
            else:
            
                dydx[i] = -Ry*(x[i]-self.BB_Prof[5])/(Rx**2.0) * 1/np.sqrt(np.abs(1.0 - ((x[i] - self.BB_Prof[5])/Rx)**2.0))
        
        return dydx
        
        
        
    def SB_profile(self,z):
        #assumes z is between WL*HSBOA  and 0.0
        #returns x position of trailing edge of SB
        if z >= self.Hsb*self.WL*self.HSBOA:
           # print(z)
           # print((1- ((z-self.Hsb*self.WL*self.HSBOA)/self.SB_Prof[1])**2.0))
            return self.SB_Prof[5] + np.sqrt(np.abs(1.0 - ((z-self.Hsb*self.WL*self.HSBOA)/self.SB_Prof[1])**2.0))*self.SB_Prof[4]
        else:
            return self.SB_Prof[5] + np.sqrt(np.abs(1.0 - ((z-self.Hsb*self.WL*self.HSBOA)/self.SB_Prof[2])**2.0))*self.SB_Prof[4]
        
    def halfBeam_SB(self, z, x):
        #returns the half breadth of the BB at height z and position x. x is a vector
        #assumes z is between 0 and WL*HSBOA assumes x less than SB_profile(z)
        
        if z >= self.Hsb*self.WL*self.HSBOA:
            Rz = self.SB_Prof[1]
        else:
            Rz = self.SB_Prof[2]
        
        Ry = np.sqrt(np.abs(1.0 - ((z - self.Hsb*self.WL*self.HSBOA)/Rz)**2.0))*self.SB_Prof[3]
        
        Rx = self.SB_profile(z) - self.SB_Prof[5]
        
        y = np.zeros((len(x),))
        
        for i in range(0,len(x)):
            
            if x[i] <= self.SB_Prof[5]:
                y[i] = Ry
            else:
                y[i] = np.sqrt(np.abs(1.0 - ((x[i] - self.SB_Prof[5])/Rx)**2.0))*Ry 
        
        return y
    
    def SB_dydx(self,z,x):
        #This function computes the slope dy/dx slope of the bulbous bow at height z and position x. This assumes x is within the bulbous bow x-range
        # x is a vector
        if z >= self.Hsb*self.WL*self.HSBOA:
            Rz = self.SB_Prof[1]
        else:
            Rz = self.SB_Prof[2]
        
        Ry = np.sqrt(np.abs(1.0 - ((z - self.Hsb*self.WL*self.HSBOA)/Rz)**2.0))*self.SB_Prof[3]
        
        Rx = self.SB_profile(z) - self.SB_Prof[5]
        
        dydx = np.zeros((len(x),))
        
        for i in range(0, len(x)):
        
            if x[i] <= self.SB_Prof[5]:
                dydx[i] =  0.0
            
            else:
                dydx[i] = -Ry*(x[i]-self.SB_Prof[5])/(Rx**2.0) * 1/np.sqrt(np.abs(1.0 - ((x[i] - self.SB_Prof[5])/Rx)**2.0)) 
            
        return dydx
        
 
        
        
    
    def gen_waterline_bow_BB(self, z, NUM_POINTS = 100, X = [0,1], bit_spaceOrGrid = 1):
        #this function returns a set of [X,Y] points that accounts for the shape and fillet radius of a bulbous bow on the bow profile
        
        a = NUM_POINTS        
        if z >= self.WL:       
            #If z is above the Ship's waterline, then the bulbous bow does not exist in that section
            return self.gen_waterline_bow(z, NUM_POINTS = a, X = X, bit_spaceOrGrid = bit_spaceOrGrid)
        
        else:
            PROF = self.solve_waterline_bow(z)
            
            x1 = self.BB_profile(z)
        
            x2 = self.delta_bow(z)
            
            
            #Set x based on spacing or grid
            if bit_spaceOrGrid:
            
                x = np.linspace(x1,x2,NUM_POINTS)
                XY = np.zeros((len(x),2))
            
            else:
                x = [i for i in X if (i > x1 and i <= x2)]
                
                x = np.concatenate(([x1],x))
                
                XY = np.zeros((len(x),2))
            
                
                
            #Find most likely point where BB intersects Bow Curve
            A = PROF.copy()
            
            A[3] = A[3] - self.halfBeam_BB(z, [self.BB_Prof[5]])
            
            
            ROOTS = np.roots(A)
            
            x_int = np.amin(np.real([i for i in ROOTS if (i >= self.bow_profile(z) and i >= self.BB_profile(z))])) #Need to call np.real as there will be instances where 2nd and 3rd roots of PROF will be imaginary. calling np.real to clean this up 
            
            
            '''
            #ind start and ending points for Rbb -> not quite a circular radius, but it is a cubic fillet (sorta counts)
             Xrad are the points forward and aft where fillet will start. 
            Xrad[0] is Rbb fraction of distance between interesect and fwd profie of BB at z 
            Xrad[1] is Rbb fraction of the distance bewtween the insect and delta_bow(z)
            '''
            dx = abs(np.amin([x_int - self.BB_profile(z), self.delta_bow(z) - x_int])) #Distance over which fillet occurs # Need to add abs to avoid dumb errors
            
            
            
            Xrad = [(x_int - dx*self.Rbb/2.0), (x_int + dx*self.Rbb/2.0)]
            
            
            # going to build quartic systems of Eqn to solve. only tricky thing: what is dydx at Xrad[0]
            
            Yrad = [self.halfBeam_BB(z, [Xrad[0]])[0], self.halfBeam_Bow([Xrad[1]], PROF)[0]]    
            dydx = [self.BB_dydx(z, [Xrad[0]])[0], self.bow_dydx([Xrad[1]], PROF)[0]]
            
            
            '''
            Rbb is quartic systems of eqns
            5 boundary conditions:
                both (Xrad, Yrad) on fillet curve
                both dydx are matched at ends
                dydx halfway between BCs is mean of BC dydx and avg slope of BC end points
            
            '''
            
            #dydx_mean = (2.0*((Yrad[1] - Yrad[0])/(Xrad[1]-Xrad[0])) + dydx[0] + dydx[1])/4.0
            
            Arad = np.array([[Xrad[0]**4.0,     Xrad[0]**3.0,   Xrad[0]**2.0,       Xrad[0],    1.0],
                             [Xrad[1]**4.0,     Xrad[1]**3.0,   Xrad[1]**2.0,       Xrad[1],    1.0],
                             [4.0*Xrad[0]**3.0, 3.0*Xrad[0]**2.0, 2.0*Xrad[0],      1.0,        0.0],
                             [4.0*Xrad[1]**3.0, 3.0*Xrad[1]**2.0, 2.0*Xrad[1],      1.0,        0.0],
                             [12.0*Xrad[0]**2.0, 6.0*Xrad[0]**2.0, 2.0,             0.0,        0.0]])
 
            
              
            brad = np.array([Yrad[0], Yrad[1], dydx[0], dydx[1], 0.0]) #dydx_mean])
            
            
            PROFrad = np.linalg.solve(Arad,brad)
                
            XY[0,:] = [x1, 0.0]
            
            xbb = [i for i in x if (i > x1 and i <= Xrad[0])]
            
            xbbrad = [i for i in x if (i > Xrad[0] and i < Xrad[1])]
            
            xbow = [i for i in x if (i >= Xrad[1] and i <= x2)]
            
            ybb = self.halfBeam_BB(z, xbb)
            
            ybbrad = np.zeros((len(xbbrad),))
            
            for i in range(0,len(xbbrad)):
                ybbrad[i] = PROFrad[0]*xbbrad[i]**4.0 + PROFrad[1]*xbbrad[i]**3.0 + PROFrad[2]*xbbrad[i]**2.0 + PROFrad[3]*xbbrad[i] + PROFrad[4]
            
            ybow = self.halfBeam_Bow(xbow, PROF)
            
        
            
            XY[1:,0] = x[1:]
            
            XY[1:,1] = np.concatenate((ybb,ybbrad,ybow)) 
                
                
            return XY


    def gen_waterline_stern_SB(self, z, NUM_POINTS = 100, X = [0,1], bit_spaceOrGrid = 1):
        #this function returns a set of [X,Y] points that accounts for the shape and fillet radius of a bulbous bow on the bow profile
        
        a = NUM_POINTS        
        if z >= self.WL*self.HSBOA:        #If z is above the Ship's waterline, then the bulbous bow does not exist in that section
            return self.gen_waterline_stern(z, NUM_POINTS = a, X=X, bit_spaceOrGrid=bit_spaceOrGrid)
        
        else:
            #Set up half beam for stern at z
            PROF = self.solve_waterline_stern(z)
            
            x1 = self.delta_stern(z)
            
            x2 = self.SB_profile(z)
            
            #Create x distribution       
            if bit_spaceOrGrid:           
                x = np.linspace(x1,x2,NUM_POINTS)
                XY = np.zeros((len(x),2))
            else:
                x = [i for i in X if (i >= x1 and i < x2)]
                x = np.concatenate((x,[x2]))
                XY = np.zeros((len(x),2))
            
            
            # This y intersect is most likely place that y in 
            y_int = self.halfBeam_SB(z, [self.SB_Prof[5]])[0] #ad the [0] so that y_int is interpretted as a float
            
            if y_int >= self.halfBeam_Stern([self.delta_stern(z)], PROF):
                y_int = self.halfBeam_Stern([self.delta_stern(z)], PROF)
                x_int = self.SB_Prof[0]
            
            else:
                if PROF[0]:
                    x_int = PROF[4]*np.sqrt(abs(1 - ((y_int-PROF[7])/PROF[5])**2.0)) + PROF[6]
                else:
                    ROOTS = np.roots([PROF[1], PROF[2], PROF[3]-y_int])

                    x_int = np.amax(np.real([i for i in ROOTS if (i >= self.delta_stern(z))]))
                
            
            #print(x_int)
            
            dx = abs(min([x_int - self.delta_stern(z), self.SB_profile(z) - x_int])) #Distance over which fillet occurs
            
            Xrad = [(x_int - dx*self.Rsb/2.0), ((x_int + dx*self.Rsb/2.0))]
            
           
            # Parabolic Radius
            
            Yrad = [self.halfBeam_Stern([Xrad[0]], PROF)[0], self.halfBeam_SB(z, [Xrad[1]])[0]]  
            
            dydx = [self.stern_dydx([Xrad[0]], PROF)[0], self.SB_dydx(z, [Xrad[1]])[0]]
            

            
            Arad = np.array([[Xrad[0]**4.0,     Xrad[0]**3.0,   Xrad[0]**2.0,   Xrad[0],    1.0],
                             [Xrad[1]**4.0,     Xrad[1]**3.0,   Xrad[1]**2.0,   Xrad[1],    1.0],
                             [4.0*Xrad[0]**3.0, 3.0*Xrad[0]**2.0, 2.0*Xrad[0],    1.0,        0.0],
                             [4.0*Xrad[1]**3.0, 3.0*Xrad[1]**2.0, 2.0*Xrad[1],    1.0,        0.0],
                             [12.0*Xrad[1]**2.0, 6.0*Xrad[1]**2.0, 2.0,             0.0,        0.0]])
 
            
              
            brad = np.array([Yrad[0], Yrad[1], dydx[0], dydx[1], 0.0])
            
            
            PROFrad = np.linalg.solve(Arad,brad)
            
            
            xstern = [i for i in x[:-1] if (i >= x1 and i <= Xrad[0])]
            
            xsbrad = [i for i in x[:-1] if (i > Xrad[0] and i < Xrad[1])]
            
                        
            xsb = [i for i in x[:-1] if (i >= Xrad[1] and i < x2)]
            
            ystern = self.halfBeam_Stern(xstern, PROF)
            
            
            ysbrad = np.zeros((len(xsbrad),))
            
            for i in range(0,len(xsbrad)):
                
                ysbrad[i] = PROFrad[0]*xsbrad[i]**4.0 + PROFrad[1]*xsbrad[i]**3.0 + PROFrad[2]*xsbrad[i]**2.0 + PROFrad[3]*xsbrad[i] + PROFrad[4]
            
            ysb = self.halfBeam_SB(z, xsb)
            
            XY[0:-1,0] = x[0:-1]
            
            XY[0:-1,1] = np.concatenate((ystern,ysbrad,ysb))
            
            # Make sure XY[-1] is zero to create a closed mesh
            XY[-1] = [x2,0.0]
            
            return XY
        
    def plot_BulbProfiles(self):
        
        # Plot intersection points in blue
        # Plot chine pt in green
        # Plot Center of Rc and Rk in red
        # half Beam(z) in black
        
        z1 = np.linspace(0.0, self.WL, num = 200)
        z2 = np.linspace(0.0, self.WL*self.HSBOA, num = 200)
        x = np.zeros((200,2))
        for i in range(0,len(z1)):
            x[i,0] = self.BB_profile(z1[i])
            x[i,1] = self.SB_profile(z2[i]) - self.Ls-self.Lm
        
        
        fig2,ax2 = plt.subplots()
        ax2.axis('equal')
        #plt.axis([0,10,0,10])
        
        ax2.plot(x[:,0], z1, '-', color = 'blue')
        ax2.plot(x[:,1], z2,'-' ,color = 'red')  

    

                
    def BulbformConstraints(self):
        
        C = np.zeros((13,))
        
        '''
        Bulb form constraints: 
            0) BBlower z radius is less than Rk
            1) BB x radius is less than Rk
            2) BB half beam is less than the half beam of the midbody at the height of max BB beam
            3) BB is FWD of Delta Bow at 0
            4) BB is FWD of Delta Bow at Vertex of Delta Bow
            5) BB is FWD of Delta Bow at WL
            6) SB lower z radius is less than Rk
            7) SB x radius is less than Rk
            8) SB half beam is less than the half beam of the midbody at the height of max BB beam
            9) SB Height overall is less than the height of the bottom of the transom
            10) delta_stern(z = 0) is fwd of the starting position of the max width of the bulb (SB_Prof[5])
            11) delta_stern(z = WL*HSBOA) is fwd of the starting position of the max width of the bulb (SB_Prof[5])
            12) if the vertex of delta_stern is less than WL*HSBOA, then it is also fwd of the starting position of the max width of the bulb (SB_Prof[5])
        '''
        
        if self.bit_BB:
            if self.Beta == 0.0:
                C[0:3] = np.array([-1.0,
                                   -1.0,
                                   self.BB_Prof[3] - self.halfBeam_MidBody(self.BB_Prof[2])])
                    
            elif self.Rk > 0.0:
                C[0:3] = np.array([self.BB_Prof[2] - self.Rk,
                                   self.BB_Prof[3] -self.Rk,
                                   self.BB_Prof[3] - self.halfBeam_MidBody(self.BB_Prof[2])])
            
            else:
                C[0:3] = np.array([1.0,1.0,1.0])
        
            if self.DELTA_BOW[0] == 0.0:
                Zv = -1.0
            else: 
                Zv = -self.DELTA_BOW[1]/ (2.0*self.DELTA_BOW[0])
        
            if Zv >=0.0 and Zv <= self.WL:
                vert_delta_bow = (self.delta_bow(Zv) - self.BB_Prof[5])
            else:
                vert_delta_bow = -1.0
            
            C[3:6] = np.array([self.BB_Prof[5] - self.delta_bow(0.0),
                               self.BB_Prof[5] - self.delta_bow(self.WL), 
                                vert_delta_bow])
            
        else:
            C[0:6] = np.array([-1.0,-1.0,-1.0, -1.0, -1.0, -1.0])
            

        
        if self.bit_SB:
            if self.Beta == 0.0:
                C[6:10] = np.array([-1.0,
                                   -1.0,
                                   self.SB_Prof[3] - self.halfBeam_MidBody(self.SB_Prof[2]),
                                   self.WL*self.HSBOA - self.SK[1]])
                    
            elif self.Rk > 0.0:
                C[6:10] = np.array([self.SB_Prof[2] - self.Rk,
                                   self.SB_Prof[3] -self.Rk,
                                   self.SB_Prof[3] - self.halfBeam_MidBody(self.SB_Prof[2]),
                                   self.WL*self.HSBOA - self.SK[1]])
            
            else:
                C[6:10] = np.array([1.0,1.0,1.0,1.0])
            
            if self.DELTA_STERN[0] == 0.0:
                Zv = -1.0
            else: 
                Zv = -self.DELTA_STERN[1]/ (2.0*self.DELTA_STERN[0])
        
            if Zv >=0.0 and Zv <= self.WL*self.HSBOA:
                vert_delta_stern = (self.delta_stern(Zv) - self.SB_Prof[5])
            else:
                vert_delta_stern = -1.0
            
            C[10:13] = np.array([self.delta_stern(0.0) - self.SB_Prof[5],
                                self.delta_stern(self.WL*self.HSBOA) - self.SB_Prof[5],
                                vert_delta_stern])
                
                
        #If no Stern Bulb, then no constraint violations
        else:
            C[6:13] = np.array([-1.0,-1.0,-1.0, -1.0, -1.0, -1.0, -1.0])
        
        
        
        
        
        
        
        return C
    

    '''
    =====================================================================
                    Section 6: Mesh Generatation
    ======================================================================
    
    This section contains functions needed to generate a complete mesh of the hull
    as an STL  
    
    
    '''
    def gen_MeshGridPointCloud(self, NUM_WL = 51, PointsPerLOA = 501, Z = [], X = [], bit_GridOrList = 1):
        # This generates each waterline with even x and z spacing in a grid
        #Z and X assignments supercede NUM_WL and PointsPerLOA Assignments
        
        if len(Z) == 0:
            Z = np.linspace(0.0001*self.Dd,self.Dd, NUM_WL)

        if len(X) == 0:
            X = np.linspace(-self.LOA*0.5,1.5*self.LOA, 2*PointsPerLOA - 1)

        Points = []

        for i in Z:
            pts = self.gen_MeshGridWL(X,i)
            
            Points.append(pts)
            
        #returns the points if user wants a waterline structured array of points
        if bit_GridOrList:
            return Points
        
        #returns a list of points in shape (N,3) if a list is preferred
        else:
            Cloud = []
            for i in Points:
                for j in i:
                    Cloud.append(j)
                
            return Cloud

    def gen_MeshGridWL(self, X, z):

        WL = []
        
        if z == 0.0:
            if self.bit_BB:
                bow_start = self.BB_Prof[5]
            else:
                bow_start = self.Kappa_BOW*self.Lb
            
            if self.bit_SB:
                stern_end = self.SB_Prof[5]
            else:
                stern_end = self.Lm + self.Lb + self.Kappa_STERN*self.Ls 
        
            
            WL.append([bow_start,0.0,0.0])
            
            x = [i for i in X if (i > bow_start and i < stern_end)]
            
            for i in x:
                WL.append([i,0.0,0.0])
                
            WL.append([stern_end,0.0,0.0])
            
        else:
            # Now generate the remaining watelines
         
            # Figure out spacing and profiles 
            if self.bit_BB:
                BOW = self.gen_waterline_bow_BB(z, X=X, bit_spaceOrGrid=0)
            else:
                BOW = self.gen_waterline_bow(z, X=X, bit_spaceOrGrid=0)
            
            for i in range(0,len(BOW)):
                WL.append([BOW[i,0], BOW[i,1], z])
             
            X_mid = [i for i in X if (i >= self.delta_bow(z) and i < self.delta_stern(z))]
            Y_mid = self.halfBeam_MidBody(z)
            
            for i in range(0,len(X_mid)):
                WL.append([X_mid[i], Y_mid, z])
            
            if self.bit_SB:
                STERN = self.gen_waterline_stern_SB(z, NUM_POINTS = 0, X=X, bit_spaceOrGrid=0)
            else:
                STERN = self.gen_waterline_stern(z, NUM_POINTS = 0, X=X, bit_spaceOrGrid=0)
            
            for i in range(0,len(STERN)):
                WL.append([STERN[i,0], STERN[i,1], z])
                
        return np.array(WL)
     
          
    def gen_pointCloud(self, NUM_WL = 50, PointsPerWL = 300, bit_GridOrList = 0, Z = []):
        # This function generates a point cloud [[xyz]0, ..., [xyz]N] of the hull
        
        #NUM_WL = number of waterlines to generate. The total waterlines must be greater than 8 to include these heights: z = [0, 0.001*self.Dd, BKz, SKz, Dd, WL, Hbb, and Hsb]
        #PointsPerWL will be divided so that each major section of the ship (stern, midbody, and bow) each gets a proportion of the points to their total length
        # bit_GridOrList determines of the PC will be returned as a grid (shape = (NUM_WL,PointsPerWL,3)) or a list (shape = (NUM_WL*PointsPerWL,3))
        # Z is an optional array of z heights to use for the point cloud or to generate the standard. Note that if Z is not empty, that the true number of waterlines will be equal to len(Z)
        
        
        

        print(Z.shape)

        
        if len(Z) == 0:
            z = self.gen_WLHeights(NUM_WL)
        else:
            z = Z
            NUM_WL = len(z)
        
        if bit_GridOrList:
            PC = np.zeros((NUM_WL, PointsPerWL,3))
            
            
            for i in range(0,len(z)):
                
                PC[i] = self.gen_WLPoints(z[i], PointsPerWL)
        
        else:
            PC = np.zeros((NUM_WL*PointsPerWL,3))
            for i in range(0,len(z)):
                WL = self.gen_WLPoints(z[i], PointsPerWL)
                PC[PointsPerWL*i:PointsPerWL*(i+1)] = np.array(WL)
        
        return PC
            
    def gen_WLHeights(self, NUM_WL):
    
        #NUM_WL = number of waterlines to generate. The total waterlines must be greater than 8 to include these heights: z = [0, 0.001*self.Dd, BKz, SKz, Dd, WL, Hbb, and Hsb]
        
        z = np.zeros((NUM_WL,))
        
        z[0:7] = np.array([self.BK[1], self.SK[1], self.WL, self.Hbb*self.WL, self.HSBOA*self.WL, self.Hsb*self.WL*self.HSBOA, 0.001*self.Dd])
        z[7:] = np.linspace(0.0, self.Dd, NUM_WL - 7)
        
        z = np.sort(z)
        
        return z
    
    def gen_WLPoints(self, z, PointsPerWL = 300):
        #This function generates the starboard waterline half breadths
        #set up baseline first between deltabow and delta stern
        # z is a scalar, PointsPerWL is also a scalar
        
        WL = []
       
        if z == 0.0:
            if self.bit_BB:
                bow_start = self.BB_Prof[5]
            else:
                bow_start = self.Kappa_BOW*self.Lb
            
            if self.bit_SB:
                stern_end = self.SB_Prof[5]
            else:
                stern_end = self.Lm + self.Lb + self.Kappa_STERN*self.Ls 
                
            
                 
            x =  np.linspace(bow_start, stern_end, PointsPerWL) 
            for i in range(0,PointsPerWL):
                WL.append([x[i], 0.0, 0.0])
                
        else:
            # Now generate the remaining watelines
            if self.bit_BB and z <= self.WL:
                bow_start = min([self.BB_profile(z), self.bow_profile(z)])
            else:
                bow_start = self.bow_profile(z)
                
            if self.bit_SB and z <= self.HSBOA*self.WL:
                
                stern_end = max([self.SB_profile(z),self.stern_profile(z)])
            else:
                stern_end = self.stern_profile(z)
                

            WL_LOA = stern_end - bow_start
            
            #print([z,WL_LOA,stern_end, bow_start])
            
            pts_bow = abs(int((self.delta_bow(z) - bow_start)/WL_LOA * PointsPerWL)) + 1
            pts_stern = abs(int((stern_end - self.delta_stern(z))/WL_LOA * PointsPerWL)) + 1
        
            if (pts_bow + pts_stern) > PointsPerWL:
                over = pts_bow + pts_stern - PointsPerWL
                pts_mid = 0
                pts_stern = pts_stern - over
            else:
                pts_mid = PointsPerWL - pts_bow - pts_stern
        
            #print(self.delta_bow(z) - bow_start)   
            #print([z, pts_bow,pts_mid,pts_stern])
                        
            if self.bit_BB:
                BOW = self.gen_waterline_bow_BB(z, NUM_POINTS = pts_bow)
            else:
                BOW = self.gen_waterline_bow(z, NUM_POINTS = pts_bow)
            
            for i in range(0,pts_bow):
                WL.append([BOW[i,0], BOW[i,1], z])
             
            X_mid = np.linspace(self.delta_bow(z), self.delta_stern(z), pts_mid+2)
            Y_mid = self.halfBeam_MidBody(z)
            
            for i in range(0,pts_mid):
                WL.append([X_mid[1+i], Y_mid, z])
            
            if self.bit_SB:
                STERN = self.gen_waterline_stern_SB(z, NUM_POINTS = pts_stern)
            else:
                STERN = self.gen_waterline_stern(z, NUM_POINTS = pts_stern)
            
            for i in range(0,pts_stern):
                WL.append([STERN[i,0], STERN[i,1], z])
            
            
                
        return np.array(WL)
        
    
    def gen_stl(self, NUM_WL = 50, PointsPerWL = 300, bit_AddTransom = 1, bit_AddDeckLid = 0, bit_RefineBowAndStern = 0, namepath = 'Hull_Mesh'):
        # This function generates a surface of the mesh by iterating through the points on the waterlines
        
        #compute number of triangles in the mesh
        #hullTriangles = 2 * (2*PointsPerWL - 2) * (NUM_WL - 1)
        #numTriangles = hullTriangles
        transomTriangles = 0
        
        #Generate WL
        z = np.zeros((NUM_WL,))
        
        z[0] = 0.0001*self.Dd
        z[1] = 0.001*self.Dd
        z[2:] = np.linspace(0.0, self.Dd, NUM_WL - 2)
        
        z = np.sort(z)

        x = np.linspace(-self.LOA*0.5,1.5*self.LOA, 2*PointsPerWL - 1)

        if bit_RefineBowAndStern:
            # Add more points to X in the bow and stern
            
            x_sub1 = x[0:int(0.75*PointsPerWL)] + 0.5*(x[1] - x[0])
            x_sub2 = x[-int(0.75*PointsPerWL):] + 0.5*(x[1] - x[0])
            x = np.concatenate((x_sub1, x_sub2, x))
            x = np.sort(x)

            

    
        #Generate MeshGrid PC
        pts = self.gen_MeshGridPointCloud(NUM_WL = NUM_WL, PointsPerLOA = PointsPerWL, Z = z, X = x, bit_GridOrList = 1)
        
        #start to assemble the triangles into vectors of indices from pts
        TriVec = []
        
        for i in range(0,NUM_WL-1):
            
            #Find idx where the mesh grids begin to align between two rows returns a zero or 1:
                
            bow = np.argmax([pts[i][0,0],pts[i+1][0,0]])
            
            stern = np.argmin([pts[i][-1,0],pts[i+1][-1,0]])
            
           
            
            # Find index where mesh grid lines up and ends between each WL
            
            if bow:
                idx_WLB1 = 1
                idx_WLB0 = np.where(pts[i][:,0] == pts[i+1][idx_WLB1,0])[0][0]
            else:
                idx_WLB0 = 1
                idx_WLB1 = np.where(pts[i+1][:,0] == pts[i][idx_WLB0,0])[0][0]
            
            if stern:
                idx_WLS1 = len(pts[i+1]) - 2
                idx_WLS0 = np.where(pts[i][:,0] == pts[i+1][idx_WLS1,0])[0][0] 
            else:
                idx_WLS0 = len(pts[i]) - 2
                idx_WLS1 = np.where(pts[i+1][:,0] == pts[i][idx_WLS0,0])[0][0]                
            
            #check that these two are the same size:
            
            #Build the bow triangles Includes Port assignments
            
            if bow:
                TriVec.append([pts[i+1][idx_WLB1], pts[i][0], pts[i+1][0]])
                
                for j in range(0,idx_WLB0):
                    TriVec.append([pts[i+1][idx_WLB1], pts[i][j+1], pts[i][j]])

                
            
            else: 
                
                for j in range(0,idx_WLB1):
                    TriVec.append([pts[i][0],pts[i+1][j], pts[i+1][j+1]])
                    
                TriVec.append([pts[i][0],pts[i+1][idx_WLB1], pts[i][idx_WLB0]])
            
            #Build main part of hull triangles. Port Assignments
            for j in range(0, idx_WLS1-idx_WLB1):
                
                TriVec.append([pts[i][idx_WLB0+j], pts[i+1][idx_WLB1+j], pts[i+1][idx_WLB1+j+1]])
                TriVec.append([pts[i][idx_WLB0+j], pts[i+1][idx_WLB1+j+1], pts[i][idx_WLB0+j+1]]) 
            
            #Build the stern:
            if stern:

                for j in range(idx_WLS0,len(pts[i])-1):
                    TriVec.append([pts[i+1][idx_WLS1],  pts[i][j+1],pts[i][j]])
                
                TriVec.append([pts[i+1][idx_WLS1], pts[i+1][-1], pts[i][-1]])
            
            else:
                
                TriVec.append([pts[i][idx_WLS0], pts[i+1][idx_WLS1], pts[i][-1]])
                
                for j in range(idx_WLS1, len(pts[i+1])-1):
                    TriVec.append([pts[i][-1], pts[i+1][j], pts[i+1][j+1]])
            
        
        TriVec = np.array(TriVec)
       
        hullTriangles = 2*len(TriVec)
        numTriangles = hullTriangles
        
        
        
        #add triangles if there is a transom
        if bit_AddTransom:
            wl_above = len([i for i in z if i > self.SK[1]])
        
            z_idx = NUM_WL - wl_above - 1
            
            transomTriangles = 2*wl_above - 1
            
            numTriangles += transomTriangles
            
        #Add triangles if there is a deck lid (meaning the ship becomes a closed body)
        if bit_AddDeckLid:
            numTriangles += 2*len(pts[-1]) - 3
    

        HULL = mesh.Mesh(np.zeros(numTriangles, dtype=mesh.Mesh.dtype))
    
        HULL.vectors[0:len(TriVec)] = np.copy(TriVec)
        
        TriVec_stbd = np.copy(TriVec[:,::-1])
        TriVec_stbd[:,:,1] *= -1
        HULL.vectors[len(TriVec):hullTriangles] = np.copy(TriVec_stbd)
    
        # NowBuild the transom:
        if bit_AddTransom:
            
            
            pts_trans = np.zeros((wl_above+1,3))
            
            for i in range(0,len(pts_trans)):                       
                pts_trans[i] = pts[z_idx+i][-1,:]
                
           
            
            pts_tranp = np.array(pts_trans)
            
            pts_tranp[:,1] *= -1.0
            
            
            
            
            HULL.vectors[hullTriangles] = np.array([pts_trans[0], pts_trans[1], pts_tranp[1]])
                        
            for i in range(1,wl_above):
                HULL.vectors[hullTriangles + 2*i-1] = np.array([pts_trans[i], pts_trans[i+1], pts_tranp[i]])
                HULL.vectors[hullTriangles + 2*i] =     np.array([pts_tranp[i], pts_trans[i+1], pts_tranp[i+1]])
                
        
        # Add the deck lid
        if bit_AddDeckLid:
            
            #pts_Lids are starboard points on the deck
            #pts_Lidp are port points on the deck
           
            pts_Lids = pts[NUM_WL-1]
            
            pts_Lidp = np.array(pts_Lids)
            pts_Lidp[:,1] *= -1.0
            
            startTriangles = hullTriangles + transomTriangles
            
            # Points are orered so the right hand rule points the lid in positive z
            HULL.vectors[startTriangles] = np.array([pts_Lids[0], pts_Lidp[1], pts_Lids[1]])
            
            for i in range(1,len(pts_Lids)-1):              
                HULL.vectors[startTriangles + 2*i - 1] = np.array([pts_Lids[i], pts_Lidp[i], pts_Lids[i+1]])
                HULL.vectors[startTriangles + 2*i] =     np.array([pts_Lids[i+1], pts_Lidp[i],  pts_Lidp[i+1]])
            
            
        HULL.save(namepath + '.stl')
        return HULL
    
 
    
    def gen_PC_for_Cw(self, draft, NUM_WL = 51, PointsPerWL = 301):
        '''
        This code generates the Point Grid and the Inputs for the Cw prediction. 
        0) Z and X -> Translated X and Z vectors for the points used in the solver. 
            X[0] = 0, X[-1] = LOA of submerged Hull. Z[0] = -draft Z[-1] = 0
        1) WL = length of Waterline
        2) Y -> point grid of Y offsets
                
        '''
        Z = np.linspace(0.00000001*self.Dd, draft, NUM_WL)
        
        x_bow = np.zeros((len(Z),))
        x_stern = np.zeros((len(Z),))
        
        for i in range(0,len(Z)):
            # Now generate the remaining watelines
            if self.bit_BB and Z[i] <= self.WL:
                x_bow[i] = self.BB_profile(Z[i])
            else:
                x_bow[i] = self.bow_profile(Z[i])
                
            if self.bit_SB and Z[i] <= self.HSBOA*self.WL:
                
                x_stern[i] = self.SB_profile(Z[i])
            else:
                x_stern[i] = self.stern_profile(Z[i])
                
        
        WL = x_stern[-1] - x_bow[-1]
        
        X = np.linspace(np.amin(x_bow), np.amax(x_stern), PointsPerWL)
        
        Y = np.zeros((PointsPerWL,NUM_WL))
        
        points = self.gen_MeshGridPointCloud(Z = Z, X = X, bit_GridOrList = 1)
        
        
        
        for i in range(0,len(Z)):
            idx = np.where(X == points[i][1][0])[0][0] #points[i,1,0] = first X in points where y != 0
            
            
            
            for j in range(1,len(points[i])-1):
                Y[idx+j-1,i] = points[i][j][1]
            
            
        X = X - X[0] # Normalize so that X[0] = 0
        Z = Z - Z[-1] # Normalize so that Z[-1] = 0
        
        return X,Z,Y,WL
    
        
        
   

        
        
    def input_Constraints(self):
        
        return np.concatenate((self.GenralHullformConstraints(),self.CrossSectionConstraints(), self.BowformConstraints(), self.SternformConstraints(), self.BulbformConstraints()))
        
      
        
    '''
    =========================================================================
                Section 7: Geometric and Volumetric Analysis Fucntions
    ==========================================================================
    
    This section contains functions to perform Naval Architecture related analyis on the geometry of the hull
    
    0) Displacement(z)        -> calculates the submerged volume at a given height, z
    1) CentersOfBuoyancy(z)   -> Returns the centers of Buoyancy for a given waterline height, z
    1) WaterplaneArea(z)      -> caluclates the area of the waterplane at height, z
    2) WaterplaneMoments(z)   -> calculates the Center of Flotation, Ixx, and Iyy of the Waterplane at height(z)
    3) MTC(z)                 -> Not Implemented yet
    4) Righting Moment(z)     -> Not Implemented Yet
    5) Block Coefficient      -> Not Implemented yet
    6) Draft, Heel and Trim   -> Not Implemented Yet
    7) LOA_wBulb              -> Returns the maximum length of the hull including added lengths from bulbs
    8) Max_Beam_midship       -> Returns the maximum beam of the midship section (calculated from midship section functions)
    9) Max_Beam_PC            -> Returns the maximum beam of the hull (Estimated from point cloud for volume calculations)
    
    '''
    
    def Calc_VolumeProperties(self, NUM_WL = 101, PointsPerWL = 1000):
        #This function generates a point cloud to be used for volumetric measurements an calls all of the evaluation measurements to be calculated. 
        Z = np.linspace(0.00001,self.Dd, num=NUM_WL)
        
        self.PCMeasurement = self.gen_pointCloud(NUM_WL = NUM_WL, PointsPerWL = PointsPerWL, bit_GridOrList = 1, Z = Z)
        
        self.Calc_WaterPlaneArea()
        
        self.Calc_Volumes()
        
        self.Calc_LCFs()
        
        self.Calc_CB(Z)
        
        self.Calc_2ndMoments()
        
        self.Calc_WettedSurface(Z)
        
        self.Calc_WaterlineLength()
        
        return Z
            
    
    def Calc_Volumes(self):
        # this Function calculates the 3D Volume of a  hull  below a height z by integrating the waterplane areas of each waterline below z as well.
        Vol = np.zeros((len(self.PCMeasurement),))
                        
        Vol[0] = 0.5*self.Areas_WP[0]*self.PCMeasurement[0,0,2]
        
        for i in range(1,len(Vol)):
            Vol[i] = Vol[i-1] +  0.5*(self.Areas_WP[i] + self.Areas_WP[i-1])*(self.PCMeasurement[i,0,2] - self.PCMeasurement[i-1,0,2])
    
        self.Volumes =  Vol
    
    def Calc_WaterPlaneArea(self):
        # This function calcuates the waterplane area for a given z height
        Areas = np.zeros((len(self.PCMeasurement),))
        
        for i in range(0,len(Areas)):
            
            WL = self.PCMeasurement[i]
            Areas[i] = 2.0*np.trapz(WL[:,1], x=WL[:,0]) 
        
        self.Areas_WP = Areas
    
    def Calc_LCFs(self):
        #This function calculates the Longitudinal Center of Flotation for the Waterplane sections
        LCF = np.zeros((len(self.PCMeasurement),))
        
        for i in range(0,len(LCF)):
            
            Moment = 0
            
            for j in range(1,len(self.PCMeasurement[i])):
                    #sum up trapezoid moments of area
                    
                 
                    dx =  self.PCMeasurement[i,j,0] - self.PCMeasurement[i,j-1,0]
                    a = 2.0 * self.PCMeasurement[i,j,1]
                    b = 2.0 * self.PCMeasurement[i,j-1,1]
                    
                    
                    
                    cx = self.PCMeasurement[i,j-1,0] + dx/3.0 * (2.0*a + b) / (a + b)
        
                    Moment = Moment + cx * 0.5*(a+b)*dx
            
            LCF[i] = Moment/self.Areas_WP[i]
        
        self.LCFs = LCF
        
    def Calc_CB(self,Z):
        #This function calculates the longitudinal and vertical centers of buoyancy for each of the waterlines
        #CB[:,0] provides the LCB and CB[:,1] is the VCB for eah volume
        
        
        CB = np.zeros((len(self.PCMeasurement),2))
        
        #Calculate Moments for X and Z directions for the Center of Buoynacy
        
        MomentX = np.multiply(self.LCFs, self.Areas_WP)
        
        MomentZ = np.multiply(Z,self.Areas_WP)
        
        
        for i in range(0,len(CB)):
        
            CB[i,0] = np.trapz(MomentX[0:i+1],x = Z[0:i+1])/self.Volumes[i]
            CB[i,1] = np.trapz(MomentZ[0:i+1],x = Z[0:i+1])/self.Volumes[i]
        
        self.VolumeCentroids = CB
    
        
    def Calc_2ndMoments(self):
        #this function calculates the second moment of area Ixx and Iyy for each WaterPlane in the form I[i] = [Ixxi,Iyyi]
            
        I = np.zeros((len(self.PCMeasurement),2))
            
        for i in range(0,len(I)):
                
            Ixx = 0.0
            Iyy = 0.0
                
            for j in range(1,len(self.PCMeasurement[i])):
                #sum up trapezoid moments of area
                dx = self.PCMeasurement[i,j,0] - self.PCMeasurement[i,j-1,0]
                a = 2.0 * self.PCMeasurement[i,j,1]
                b = 2.0 * self.PCMeasurement[i,j-1,1]
                    
                cx = self.PCMeasurement[i,j-1,0] + dx/3.0 * (2.0*a+b)/(a+b)
                    
                Ixx = Ixx + dx/48.0 * (a + b) * (a**2.0 + b**2.0)
                    
                Iyy = Iyy + dx**3.0 * (a**2.0 + 4.0*a*b + b**2.0)/(36*(a+b)) + 0.5*(a+b)*dx*(self.LCFs[i] - cx)**2.0
                
            I[i] = [Ixx,Iyy]
                
        self.I_WP = I

    def Calc_WettedSurface(self,Z):
        # This function calcultes and summates the wetted surface between each draft line (Z[]and at the bottom of the hull, by estimating length along the surface
        
        ArcL = np.zeros((len(Z),))
        WSA = np.zeros((len(Z),))
    
        
        for i in range(0,len(ArcL)):
            
            for j in range(1,len(self.PCMeasurement[0])):
                
                #dL = distance of length along outside of ship along waterline at z[i]
                dL = np.sqrt((self.PCMeasurement[i,j,0] - self.PCMeasurement[i,j-1,0])**2.0 + (self.PCMeasurement[i,j,1] - self.PCMeasurement[i,j-1,1])**2.0)
                
                ArcL[i] = ArcL[i] + dL
                
            ArcL[i] + ArcL[i] + self.PCMeasurement[i,-1,1] #Add transom width to Arc Length
            
        #Wetted Surface Area is integral of Arc Length from 0 to Z[idx]
        
        WSA[0] = 2*self.Areas_WP[0] #wetted surface area at bottom of hull is approximately area of waterplane at z ~=0
        
        for i in range(1,len(Z)):
         
         #Wetted Surface area is cumulative sum of WSA at each height (2*0.5 for trapezoid rule and for 2 sides of hull = 1)
         WSA[i] = WSA[i-1] +  (ArcL[i]+ArcL[i-1])*(Z[i] - Z[i-1])
                
        self.Area_WS = WSA
        
        
    def Calc_WaterlineLength(self):
        #This function returns the length of the waterline for each Z
        
        WLL = np.zeros((len(self.PCMeasurement),))
        
        for i in range(0,len(WLL)):
            #Length of Stern Position - Bow Position in X
            WLL[i] = self.PCMeasurement[i,-1,0] - self.PCMeasurement[i,0,0]
        
        self.WL_Lengths = WLL

    def Calc_LOA_wBulb(self):
        #This function returns the length of the hull including the bulb lengths
        
        if self.bit_BB:
            bow_start = min([0.0,self.BB_Prof[5]-self.BB_Prof[4]])
        else:
            bow_start = 0.0
        
        if self.bit_SB:
            stern_end = max([self.LOA, self.SB_Prof[5]+self.SB_Prof[4]])
        else:
            stern_end = self.LOA
        
        self.LOA_wBulb = stern_end - bow_start
        
        return self.LOA_wBulb
    
    def Calc_Max_Beam_midship(self):
        #This function returns the maximum beam of the midship section (calculated from midship section functions)

        #fist check Bd vs Bc
        if self.Bd >= self.Bc:
            self.Max_Beam_midship = self.Bd*2.0
        else:
            self.Max_Beam_midship = (self.Rc_Center[0] + self.Rc)*2.0 #Max beam is the y coordinate of the center of the chine plus the radius of the chine

        return self.Max_Beam_midship
    
    def Calc_Max_Beam_PC(self):
        #This function returns the maximum beam of the hull (Estimated from point cloud for volume calculations)
        
        self.Max_Beam_PC = 2.0*np.amax(self.PCMeasurement[:,:,1])
        
        return self.Max_Beam_PC
            

    def interp(A,Z,z):
        # This function interpolates data to approximate A(z)  given values of A(Z) 
    
        idx = np.where(Z < z)[0][-1]
    
        frac = (z - Z[idx])/(Z[idx+1] - Z[idx])
    
        return A[idx] + frac*(A[idx+1] - A[idx])


