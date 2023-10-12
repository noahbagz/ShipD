# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 20:17:37 2022

@author: nbagz

This code is based off of sample code provided in the PhD Defense of Douglas Read from the University of Maine (2009):
    
    'A Drag Estimate for Concept Stage Ship Design Optimization'
    
    https://digitalcommons.library.umaine.edu/cgi/viewcontent.cgi?referer=&httpsredir=1&article=1509&context=etd

    Source Code Date of Publication is July 30 2008

The goal of this code is to use Michell Thin Ship Theory to predict the Wave Drag from a ship progressing at a steady speed U

The inputs to this code are:
    Y -> The y offsets of a Ship hull at scale, such that they form a point grid of the offsets
        INDEXING [X_idx,Zidx]
    U -> The speed of the hull
    X -> Vector of X Positions on the Ship -> Must be odd in length -> uniform spacing
    Z -> Vecotr of Z positions from 0 to -Draft that are uniformly distributed
    RHO-> Density of water/fluid
    N -> Number of Angles to measure wave propagation from for numerical integration
    

THE FOLLOWING CODE IS MODIFIED BY NOAH JOSEPH BAGAZINSKI in NOVEMBER 2022 to Translate the source code from matlab to python



"""
import numpy as np

g = 9.81 #set up gravity Constant

def ModMichell(Y,U,X,Z,RHO,N):
    
    Nx = len(Y) # determine number of stations
    Nz = len(Y[0]) # determine number of waterlines
    #YH = np.multiply(Y,B) #scale non-dimensional offsets by the beam
    if Nx % 2 == 0:
        print('Nx must be odd.') #required for x Filon algorithm
        return -1
    # ------ integration variables ---------
    dz = Z[1] - Z[0]
    
    
    L = X[-1]
    
    dx = L/(Nx-1)
    
    
    #theta = linspace(0,pi/2,N); theta = theta’;
    theta = michspace(N)
    

    k0 = g/U**2.0               # fundamental wave number
    c = (4.0*RHO*U**2)/np.pi    # constant
    a = 1.0/np.cos(theta)         # convenient substitution a = sec(theta)
    k = k0*a**2.0 # dispersion relation
    
    
    #---------- Z INTEGTRAL ----------------
    # --- variables for Filon trapezoidal algorithm ---
    
    Kz = k0*dz*a**2.0
    
    Kzsq = Kz**2.0
    
    w0 = (np.exp(Kz)-1.0-Kz)/Kzsq   #Note : np.expm1(x)  = (e^x - 1)
    wn = (np.exp(Kz)-1.0 + np.exp(-Kz)-1.0)/Kzsq
    wN = (np.exp(-Kz)-1.0 + Kz)/Kzsq
    
    # --- preallocate ---
    f = np.zeros((Nz,))
    F = np.zeros((Nx,N))

    for j in range(0,N):
        for m in range(0,Nx):
            for n in range(0,Nz):
                
                if n == 0:
                    f[n] = w0[j]*Y[m,n]*np.exp(k0*Z[n]*a[j]**2.0)*dz
                    
                elif n == Nz-1:
                    f[n] = wN[j]*Y[m,n]*np.exp(k0*Z[n]*a[j]**2.0)*dz
                    
                else:
                    
                    f[n] = wn[j]*Y[m,n]*np.exp(k0*Z[n]*a[j]**2.0)*dz

            F[m,j] = sum(f);

    #---------- X INTEGRAL -----------------
    #--- variables for Filon algorithm ---
    
    Kx = k0*dx*a
    alp = (Kx**2 + 0.5*Kx*np.sin(2.0*Kx) + np.cos(2*Kx) - 1)/Kx**3.0
    bet = (3.0*Kx + Kx*np.cos(2.0*Kx) - 2.0*np.sin(2*Kx))/Kx**3.0
    gam = 4.0*(np.sin(Kx) - Kx*np.cos(Kx))/Kx**3.0
    
    Nev = int((Nx+1)/2)# even Filon index
    Nod = int((Nx-1)/2) # odd Filon index
    
    # --- preallocate ---
    pev = np.zeros((Nev,)) 
    qev = np.zeros((Nev,))
    pod = np.zeros((Nod,)) 
    qod = np.zeros((Nod,))
    
    Pt = np.zeros((N,)) 
    Qt = np.zeros((N,)) 
    Pev = np.zeros((N,)) 
    Qev = np.zeros((N,)) 
    Pod = np.zeros((N,)) 
    Qod = np.zeros((N,)) 
    P = np.zeros((N,)) 
    Q = np.zeros((N,)) 
    
    
    for j in range(0,N):
        for m in range(0,Nev):
            pev[m] = F[2*m,j]*np.cos(k0*X[2*m]*a[j])
            qev[m] = F[2*m,j]*np.sin(k0*X[2*m]*a[j])

        for m in range(0,Nod):
            pod[m] = F[2*m+1,j]*np.cos(k0*X[2*m+1]*a[j])
            qod[m] = F[2*m+1,j]*np.sin(k0*X[2*m+1]*a[j])

        Pt[j] = F[-1,j]*np.cos(k0*L*a[j])
        Qt[j] = F[-1,j]*np.sin(k0*L*a[j])
                  
        Pev[j] = sum(pev)-0.5*Pt[j]
        Pod[j] = sum(pod)
        Qev[j] = sum(qev)-0.5*Qt[j]
        Qod[j] = sum(qod)
        
        P[j]=dx*( alp[j]*Qt[j]+bet[j]*Pev[j]+gam[j]*Pod[j])
        Q[j]=dx*(-alp[j]*Pt[j]+bet[j]*Qev[j]+gam[j]*Qod[j])

    R = c*k**2.0/(a**3.0) * (k**2.0 * ( P**2.0 + Q**2.0) + 2.0*k*a*(Q*Pt - P*Qt) + a**2.0 * (Pt**2.0 + Qt**2.0))
    
    R = np.nan_to_num(R)
    
    #---------- THETA INTEGRAL -------------

    rw = np.zeros((N-1,))
    
    for k in range(0,N-1):
        rw[k] = 0.5*(R[k]+R[k+1])*(theta[k+1]-theta[k])
        
    return sum(rw)
    #------------ END ----------------------



def michspace(N):
    '''    
    MICHSPACE log spacing for Michell integral

     MICHSPACE(N) produces log base 10 spacing over N propagation
     angles between 0 and pi/2. Points are more closely spacednear pi/2.
    '''
  
    xm = (np.logspace(0,1,N, base=10.0)-1.0)*np.pi/18-np.pi/2
    xm = -xm[::-1] # flips order of array 
    
    return xm


def CalcDrag(U,LOA,WL,CW,T,SA, rho=1025.0):
    '''
    Parameters
    ----------
    U : Ship Speed in m/s
    LOA : Length Overall in m, First term in Hull Parameterization.
    WL : Waterline length in m at Draftmark T. NOTE: WL needs to be calculated as an input
    SA : Wetted Surface Area of hull in m^2.
    CW : Vector of Wave Drag Calculations. [size 32]
    T: Fraction of Dd that is draft of hull.
    SA: Wetted Surface Area of hull in m^2 at Draft Mark T. NOTE: SA needs to be calcuated as an input
    rho : density of water in kg/m^3
        rho for fresh water is  1000 kg/m^3
        rho for salt water is 1025 kg/m^3
    

    Returns
    -------
    This function will interpolate the Cw input based on U,WL, and T 
    
    Calculated total drag of Wave Drag + Skin Friction Drag
    
    '''
    
    Cf = Calc_Cf(U=U,WL=WL)
    
    Rf = 0.5*Cf*rho*SA*U**2.0
    
    
    Fn = U/np.sqrt(g*WL) # Froude Number
    
    
    Cw = interp_CW(Fn=Fn,T=T,CW=CW)
    
    Rw = 0.5*Cw*rho*(LOA**2.0)*(U**2.0)
    
    return (Rw + Rf), Fn
    

def Calc_Cf(U, WL, v = 1.19*10.0**-6.0):
    '''
    This Function Calculates the Skin Friction Coefficent Based on the 1957 ITTC Skin Friction Line.
    The inputs calculate the Reynolds number of the ship and return the Cf estimate
    
    Parameters
    ----------
    U : float scalar
        hull speed in m/s.
    WL : waterline length in m
        DESCRIPTION.
    v : kinematic viscosity of water in m^2/s
        v  = 1.14*10^-6 m^2/s for fresh water 
        v = 1.19*10^-6 m^2/s for saltwater

    
    inputs:
        U: Speed in [m/s]
        WL: Waterline length of the ship in [m]
        v: Kinematic Viscosity of water [m^2/s]
        

    Returns
    -------
    Cf : Float scalar
        1957 ITTC Estimate of  viscous skin friction drag.

    '''
    Re = U*WL/v
    
    Cf = 0.075/(np.log10(Re)-2.0)**2.0
    
    return Cf

def interp_CW(Fn,T,CW, Fn_list = np.array([0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45]), T_list = np.array([0.25,0.33,0.5,0.67])):

    '''
    This Function takes in Froude Number and Draft Mark to interpolate between these to 
    estimate the wave drag coefficient
    

    Parameters
    ----------
    Fn : scalar, Froude Number = U/sqrt(g*WL)
        
    T : Draft (As a fraction of Dd).
    CW : vector of wave drag predictions for draft marks an 

    Returns
    -------
    cw : interpolated cw prediction based on Fn and T inputs

    '''
    Cw_tab = np.reshape(CW, (len(T_list), len(Fn_list)))
    
    
    Fn_idx = 0
    Fn_frac = 0
    
    T_idx = 0
    T_frac = 0
    
    # Get positions and fractions of froude number and draft where Fn and T belong within the lists
    if Fn >= Fn_list[-1]:
        Fn_idx = len(Fn_list) - 2
        Fn_frac = 1.0
        
    elif Fn <= Fn_list[0]:
        Fn_idx = 0
        Fn_frac = 0.0
        
    else:
        arr = np.where(Fn_list < Fn)[0]
        Fn_idx = arr[-1]
        Fn_frac = (Fn-Fn_list[Fn_idx])/(Fn_list[Fn_idx+1] -Fn_list[Fn_idx])
    
    if T >= T_list[-1]:
        T_idx = len(T_list) - 2
        T_frac = 1.0
        
    elif T <= T_list[0]:
        T_idx = 0
        T_frac = 0.0
        
    else:
        arr = np.where(T_list < T)[0]
        T_idx = arr[-1]
        T_frac = (T-T_list[T_idx])/(T_list[T_idx+1] -T_list[T_idx])
    
    # Interpolate between upper and lower limits of Froude Number
    
    CwL = Cw_tab[T_idx,Fn_idx] + Fn_frac * (Cw_tab[T_idx,Fn_idx+1] - Cw_tab[T_idx,Fn_idx])
    
    CwU = Cw_tab[T_idx+1,Fn_idx] + Fn_frac * (Cw_tab[T_idx+1,Fn_idx+1] - Cw_tab[T_idx+1,Fn_idx])
    
    # Interpolate between upper and lower limits of Fn
    cw = CwL + T_frac*(CwU - CwL)   
        
    
    return cw
    
    
