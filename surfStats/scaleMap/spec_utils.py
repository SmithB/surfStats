#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 21:29:21 2023

@author: ben
"""
import numpy as np

def az_lambda(nx, ny, dx, fold=False):
    eps=float(1e-10)
    if fold is True:
        kx, ky=np.meshgrid( 2*np.pi*np.r_[ eps, np.arange(1.,nx/2+1)]/nx/dx,
                            2*np.pi*np.r_[ eps, np.arange(1,ny/2), np.arange(-ny/2,0.)]/ny/dx);
    else:
        kx, ky=np.meshgrid( 2*np.pi*np.r_[ eps, np.arange(1.,nx/2), np.arange(-nx/2,0.)]/nx/dx,
                                    2*np.pi*np.r_[ eps, np.arange(1,ny/2), np.arange(-ny/2,0.)]/ny/dx);
    L=2*np.pi/np.sqrt(kx**2+ky**2);
    L[0,0]=2*nx*dx;

    az= np.arctan2(ky, kx);
    return az, L, kx, ky

def hanning2(n): 
    x=np.arange(-(n/2-0.5), (n/2-0.5)+1);
    [x,y]=np.meshgrid(x,x);
    r=np.sqrt(x**2+y**2);

    w=0.5+0.5*np.cos(2*np.pi*r/(n+1));  
    w[r>n/2]=0.; 
    return w


def gen_cov(W, x, y, xbar=None, ybar=None, sumW=None, sumW2=None):
    if sumW is None:
        sumW=np.sum(W.ravel())
    if sumW2 is None:
        sumW2=np.sum(W.ravel()*W.ravel())
    if xbar is None:
        xbar=np.sum(W*x)/sumW
        ybar=np.sum(W*y)/sumW
    xw=W*(x-xbar)
    yw=W*(y-ybar)
    covxx=np.sum(xw*xw) 
    covyy=np.sum(yw*yw) 
    covxy=np.sum(xw*yw)
    C=np.array([[covxx, covxy], [covxy, covyy]])/sumW2
    return C
