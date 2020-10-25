# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:14:53 2019

@author: baccarini_a
"""

def Particle_DiffCoeff(T,Cc,Dp):
    """This function calculates the diffusion coefficient for an
    aerosol particle with a diameter Dp. This script is based on
    the Stokes-Einstein equation. Viscosity is scaled based on the 
    Sutherland equation.
    Input:
        T in kelvin
        Cc is the Cunningham slip correction factor
        Dp in nanometer
    Output:
        Diffusion coefficient in cm2/s"""
    #viscosity calculation
    eta=1.458e-6*T**1.5/(T+110.4)
    
    Dp=Dp*10**(-9) #convert particle diameter in meters
    Kb=1.3806e-23
    D=Kb*T*Cc/(3*np.pi*eta*Dp)*10**4
    
    return D
    
    
def Cunningham(Dp,P,T):
    """This function calculates the Cunningham slip correction factor
    based on the parametrization from Kim et al 2005.
    Input:
        Dp particle diameter in nm
        P in hPa
        T in kelvin
    Output:
        Cc"""
    #Air mean free path calculation
    T0=296.15
    P0=1013.25
    lambd_zero=67.3 #reference mean free path at std conditions
    
    lambd=lambd_zero*(T/T0)*(P0/P)*((1+110.4/T0)/(1+110.4/T))
    
    Kn=2*lambd/Dp #Knudsen number
    
    Cc=1+Kn*(1.165+0.483*np.exp(-0.997/Kn))
    
    return Cc