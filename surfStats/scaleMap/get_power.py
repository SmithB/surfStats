#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 21:30:47 2023

@author: ben
"""
import numpy as np
import time
from .spec_utils import gen_cov

try:
   import pyfftw as fft
   #fft.config.NUM_THREADS=1
   #fft.interfaces.cache.enable()
except ImportError:
    import numpy.fft as fft

def get_power(img, W, lambda_els, kx, ky, use_fftw=False, Wsum=1, Wsum2=1, use_mean=False):
    Norm=np.sum(W.ravel());
    if Norm>0:
        Norm=1./Norm
    start=time.time()
    if use_fftw:
        if len(img.shape)<=2:
            fft_buffer=fft.empty_aligned(img.shape)
        if len(img.shape)==3:
            fft_buffer=fft.empty_aligned(img.shape[1:])

    if len(img.shape)<=2 or img.shape[0]==1:
        # this is the image case or the x-gradient-only case
        bar=np.sum((img*W).ravel())*Norm
        if use_fftw:
            fft_buffer[:]=W*(img-np.mean(img.ravel()))
            P_IMG=np.abs(fft.interfaces.numpy_fft.rfftn(fft_buffer))**2
        else:
            P_IMG=np.abs(fft.rfftn((img-np.mean(img.ravel()))*W))**2
    else:
        # this is the isotropic case: img is a list that holds the x and y slopes
        bar=np.sum((np.abs(img[0,:,:])+np.abs(img[1,:,:])).ravel()*W.ravel())*Norm
        if use_fftw:
            fft_buffer[:]=W*(img[0,:,:]-np.mean(img[0,:,:].ravel()))
            P_IMG=0.5*np.abs(fft.interfaces.numpy_fft.rfftn(fft_buffer, overwrite_input=False))**2
            fft_buffer[:]=W*(img[1,:,:]-np.mean(img[1,:,:].ravel()))
            P_IMG=P_IMG+0.5*np.abs(fft.interfaces.numpy_fft.rfftn(fft_buffer, overwrite_input=False))**2
        else:
            P_IMG=0.5*(np.abs(fft.rfftn((img[0,:,:]-np.mean(img[0,:,:].ravel()))*W))**2.+np.abs(fft.rfftn((img[1,:,:]-np.mean(img[1,:,:].ravel()))*W))**2.)
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
        e_vals,e_vecs=np.linalg.eig(CC)
        maxev=np.argmax(e_vals)
        minev=1-maxev
        az[ii]=180./np.pi * np.arctan2(e_vecs[0, maxev], e_vecs[1, maxev])
        if e_vals[minev]==0:
            R[ii]=np.nan
        else:
            R[ii]=np.sqrt(e_vals[maxev]/e_vals[minev])
    return P, az, R, bar, fft_time
