#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 21:30:47 2023

@author: ben
"""
import numpy as np
import time
from .spec_utils import gen_cov

import numpy.fft as npfft

# pyfftw is optional.  Note that the FFT entry points live in
# pyfftw.interfaces.numpy_fft, not on the pyfftw top level, so the two backends
# have to be kept under separate names.
try:
   import pyfftw
   import pyfftw.interfaces.numpy_fft as fftw
except ImportError:
    pyfftw = None
    fftw = None

def get_power(img, W, lambda_els, kx, ky, use_fftw=False, Wsum=1, Wsum2=1, use_mean=False):
    if use_fftw and pyfftw is None:
        use_fftw=False
    img=np.asarray(img)
    if img.ndim==3 and img.shape[0]==1:
        # a single-band stack is the plain image case; drop the leading axis so
        # that the rFFT is taken over two dimensions rather than three
        img=img[0,:,:]
    Norm=np.sum(W.ravel());
    if Norm>0:
        Norm=1./Norm
    start=time.time()
    if use_fftw:
        if img.ndim<=2:
            fft_buffer=pyfftw.empty_aligned(img.shape)
        else:
            fft_buffer=pyfftw.empty_aligned(img.shape[1:])

    if img.ndim<=2:
        # this is the image case or the x-gradient-only case
        bar=np.sum((img*W).ravel())*Norm
        if use_fftw:
            fft_buffer[:]=W*(img-np.mean(img.ravel()))
            P_IMG=np.abs(fftw.rfftn(fft_buffer))**2
        else:
            P_IMG=np.abs(npfft.rfftn((img-np.mean(img.ravel()))*W))**2
    else:
        # this is the isotropic case: img is a list that holds the x and y slopes
        bar=np.sum((np.abs(img[0,:,:])+np.abs(img[1,:,:])).ravel()*W.ravel())*Norm
        if use_fftw:
            fft_buffer[:]=W*(img[0,:,:]-np.mean(img[0,:,:].ravel()))
            P_IMG=0.5*np.abs(fftw.rfftn(fft_buffer, overwrite_input=False))**2
            fft_buffer[:]=W*(img[1,:,:]-np.mean(img[1,:,:].ravel()))
            P_IMG=P_IMG+0.5*np.abs(fftw.rfftn(fft_buffer, overwrite_input=False))**2
        else:
            P_IMG=0.5*(np.abs(npfft.rfftn((img[0,:,:]-np.mean(img[0,:,:].ravel()))*W))**2.+np.abs(npfft.rfftn((img[1,:,:]-np.mean(img[1,:,:].ravel()))*W))**2.)
    fft_time=time.time()-start
    P_IMG=P_IMG.ravel()
    az=np.zeros([len(lambda_els),1])
    P=np.zeros([len(lambda_els),1])
    R=np.zeros([len(lambda_els),1])
    for ii, these in enumerate( lambda_els):
        if use_mean is True:
            P[ii]=np.mean(P_IMG[these])
        else:
            P[ii]=np.sum(P_IMG[these])
        if P[ii]==0:
            P[ii]=np.nan
        CC=gen_cov(P_IMG[these], kx[these], ky[these], xbar=0., ybar=0., sumW=Wsum, sumW2=Wsum2)
        if np.any(np.isnan(CC.ravel())):
            # NaNs cause eig to crash
            az[ii]=np.nan
            R[ii]=np.nan
            continue
        # CC is symmetric, so eigh is both correct and cheaper than eig, and it
        # returns real eigenvalues in ascending order
        e_vals,e_vecs=np.linalg.eigh(CC)
        minev, maxev = 0, 1
        az[ii]=180./np.pi * np.arctan2(e_vecs[0, maxev], e_vecs[1, maxev])
        if e_vals[minev]<=0 or e_vals[maxev]<0:
            # a non-positive eigenvalue is numerical noise on a degenerate
            # covariance; the ratio is not meaningful
            R[ii]=np.nan
        else:
            R[ii]=np.sqrt(e_vals[maxev]/e_vals[minev])
    return P, az, R, bar, fft_time
