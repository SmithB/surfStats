#! /usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:41:53 2013

@author: ben
"""

try:
   from osgeo import gdal, gdalconst
except ImportError:
   import gdal
   from gdal import gdalconst
   
from multiprocessing import Pool, freeze_support
import argparse
#import scipy.fftpack as sfft
import scipy.ndimage as snd
import scipy.interpolate as sint
import numpy as np
import matplotlib.pyplot as plt
from im_subset import im_subset
try:
   import pyfftw as fft
except ImportError:
    import numpy.fft as fft
#import numpy.fft as fft    
import sys, os
import time
np.seterr(invalid='ignore')

def az_lambda(nx, ny, dx, fold=False):
    eps=np.float(1e-10)
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

def get_P_wrapper(argList):
    [ img, W, lambda_els, kx, ky, use_fftw, Wsum, Wsum2, use_mean, r_out, c_out]=argList
    P, az, R, bar, fft_time=get_P(img, W, lambda_els, kx, ky, use_fftw=use_fftw, Wsum=Wsum, Wsum2=Wsum2, use_mean=use_mean)
    return (r_out, c_out, P, az, R, bar, fft_time)
    
def get_P(img, W, lambda_els, kx, ky, use_fftw=False, Wsum=1, Wsum2=1, use_mean=False):
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
            P_IMG=0.5*np.abs(fft.interfaces.numpy_fft.rfftn(fft_buffer, overwrite_input=False, threads=1))**2
            fft_buffer[:]=W*(img[1,:,:]-np.mean(img[1,:,:].ravel()))
            P_IMG=P_IMG+0.5*np.abs(fft.interfaces.numpy_fft.rfftn(fft_buffer, overwrite_input=False, threads=1))**2
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
            P[ii]=np.NaN
        CC=gen_cov(P_IMG[these], kx[these], ky[these], xbar=0., ybar=0., sumW=Wsum, sumW2=Wsum2)
        if np.any(np.isnan(CC.ravel())):
            # NaNs cause eig to crash
            az[ii]=np.NaN
            R[ii]=np.NaN
            continue
        e_vals,e_vecs=np.linalg.eig(CC)
        maxev=np.argmax(e_vals)
        minev=1-maxev
        az[ii]=180./np.pi * np.arctan2(e_vecs[0, maxev], e_vecs[1, maxev])
        if e_vals[minev]==0:
            R[ii]=np.NaN
        else:
            R[ii]=np.sqrt(e_vals[maxev]/e_vals[minev])
    return P, az, R, bar, fft_time

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


def main():
    fold=True
    parser = argparse.ArgumentParser(description='calculate scale map on the specified file.  Positional arguments give the input file and the maximum scale')
    parser.add_argument('input_file')
    parser.add_argument('N_LW')
    parser.add_argument('--erode_scale', '-e', type=float, default=None)
    parser.add_argument('--take_slope', '-t', action='store_true', default=False)
    parser.add_argument('--take_log', '-L', action='store_true', default=False)
    parser.add_argument('--use_mean','-m', action='store_true');
    parser.add_argument('--prefilter_width','-p', type=float, default=None)
    parser.add_argument('--num_processes','-n', type=int, default=1)
    parser.add_argument('--isotropic', '-i', action='store_true');
    args=parser.parse_args()
    
    thefile=args.input_file
    N_LW=int(args.N_LW)
    
    if args.num_processes>1:
        # setup the multiprocessing pool
        myPool=Pool(args.num_processes);
    # check if pyfftw is enabled, activate the cache if it is
    try:
        fft.interfaces.cache.enable()
        use_fftw=True
    except:
        use_fftw=False
    
    ds=gdal.Open(thefile);
    
    out_base=os.path.splitext(thefile)[0];
    
    if args.use_mean is True:
        out_base=out_base+"_mean"
        
    P_file=out_base+'_P_fft2.tif'
    Ps_file=out_base+'_Ps_fft2.tif'
    az_file=out_base+'_az_fft2.tif'
    R_file=out_base+'_R_fft2.tif'
    out_nodata=np.NaN
        
    print("working on %s, outfile is %s" % (thefile, P_file) )
    if args.use_mean is True:
        print("----------using the mean instead of the sum")
            
    for file in (P_file, az_file, R_file):
        if os.path.exists(file): 
            print("outfile %s exists, deleting" % file)
            os.remove(file)
  

    driver=gdal.GetDriverByName("GTiff")
    xform=np.array(ds.GetGeoTransform())

    scales=2.**np.arange(1, 1.+int(np.log2(N_LW)));
    print("running scales:")
    print(scales)

    print("NOTE:: using two times the largest scale for N")
    N=(scales[-1])*2.

 
    az, L, kx, ky=az_lambda(N, N, 1, fold=fold)
    L_bins=[np.ravel_multi_index(np.nonzero((L>=this) & (L < 2*this)), L.shape) for this in scales]
    W=hanning2(N)
    Wsum=np.sum(W.ravel())
    Wsum2=np.sum(W.ravel()*W.ravel())

    N_bands=len(scales)
    bands=1+np.arange(N_bands)
    band=ds.GetRasterBand(1)
    #N.B. changed N/2 to N/4
    dec=N/4;
    blocksize=np.minimum(4096, 8*N)
    inNoData=band.GetNoDataValue()
    if inNoData is None:
        inNoData = 0
    print("inNoData is %f" % (inNoData))
    print("stride is %d" % (blocksize-2*N))

    nX=band.XSize;
    nY=band.YSize;
    P_Ds = driver.Create(P_file, int(nX/dec), int(nY/dec), len(scales), gdalconst.GDT_Float32, options = ['BigTIFF=YES'])
    P_sub=im_subset(0, 0, int( nX/dec), int(nY/dec), P_Ds, Bands=bands)
    az_Ds = driver.Create(az_file, int(nX/dec), int(nY/dec), len(scales), gdalconst.GDT_Float32, options = ['BigTIFF=YES'])
    az_sub=im_subset(0, 0, int( nX/dec), int(nY/dec), az_Ds, Bands=bands)
    R_Ds = driver.Create(R_file, int(nX/dec), int(nY/dec), len(scales), gdalconst.GDT_Float32, options = ['BigTIFF=YES'])
    R_sub=im_subset(0, 0, int( nX/dec), int(nY/dec), R_Ds, Bands=bands)

    Ps_Ds = driver.Create(Ps_file, int(nX/dec), int(nY/dec), len(scales), gdalconst.GDT_Float32, options = ['BigTIFF=YES'])
    Ps_sub=im_subset(0, 0, int( nX/dec), int(nY/dec), Ps_Ds, Bands=bands)

    total_fft_time=0.0
    start_time=time.time()
    for out_ds in (P_Ds, az_Ds, R_Ds, Ps_Ds):
        for bandN in [bands[0]]:
            band=out_ds.GetRasterBand(int(bandN))
            band.SetNoDataValue(out_nodata)

    if args.erode_scale is not None:
        erode_kernel=np.ones([1, 2*args.erode_scale]).astype('bool')

    start=time.time()
    dtime=0
    stride=blocksize-2*N
    print("nX_out=%f, nY_out=%f" % (int(nX/dec), int(nY/dec)))
    N_out_sub=int(blocksize/dec)-1
    for in_sub in im_subset(0, 0,  nX,  nY, ds, pad_val=0, Bands=[1], stride=blocksize-2*N, pad=N, no_edges=False):
        P_sub.setBounds(int((in_sub.c0+N/2)/dec), int((in_sub.r0+N/2)/dec), N_out_sub, N_out_sub)
        P_sub.z=np.zeros([N_bands, int( N_out_sub), int(N_out_sub)])+out_nodata;
        az_sub.setBounds(int((in_sub.c0+N/2)/dec), int((in_sub.r0+N/2)/dec), N_out_sub, N_out_sub)
        az_sub.z=np.zeros([N_bands, int( N_out_sub), int(N_out_sub)])+out_nodata;
        R_sub.setBounds(int((in_sub.c0+N/2)/dec), int((in_sub.r0+N/2)/dec), N_out_sub, N_out_sub)
        R_sub.z=np.zeros([N_bands, int( N_out_sub), int(N_out_sub)])+out_nodata;

        Ps_sub.setBounds(int((in_sub.c0+N/2)/dec), int((in_sub.r0+N/2)/dec), N_out_sub, N_out_sub)
        Ps_sub.z=np.zeros([N_bands, int( N_out_sub), int(N_out_sub)])+out_nodata;

        sys.stdout.write("\r\b r0=%d/%d, c0=%d/%d, last dt=%f" %(int(in_sub.r0/stride), int(float(in_sub.Nr)/float(stride)), int(float(nY)/float(stride)),int(nX/stride), dtime))
        sys.stdout.flush()
    
        if np.all(np.logical_or(in_sub.z == 0, np.logical_or(np.isnan(in_sub.z), in_sub.z==inNoData))):
            P_sub.writeSubsetTo(bands, P_sub)
            az_sub.writeSubsetTo(bands, az_sub)
            R_sub.writeSubsetTo(bands, R_sub)
            Ps_sub.writeSubsetTo(bands, Ps_sub)
            continue
    
        if args.erode_scale is not None:
            mask=np.logical_or(in_sub.z == 0., np.logical_or(np.isnan(in_sub.z), in_sub.z==inNoData))
            if np.any(mask):
                mask[0,:,:]=snd.morphology.binary_dilation(mask[0,:,:], structure=erode_kernel)
                mask[0,:,:]=snd.morphology.binary_dilation(mask[0,:,:], structure=erode_kernel.transpose())
            if np.all(mask):
                continue
            if args.landsat:
                in_sub.z=np.maximum(0, in_sub.z*2.e-5-0.1)

            in_sub.z[mask]=inNoData
            
        if args.take_slope:
            dx=ds.GetGeoTransform()[1]
            in_sub.z=np.float32(in_sub.z)
            if args.erode_scale is None:
                mask=in_sub.z==inNoData    
            if args.prefilter_width is not None:
                #print "prefiltering with kernel of width %f" % args.prefilter_width
                in_sub.z[0,:,:]=snd.filters.gaussian_filter(in_sub.z[0,:,:], args.prefilter_width, mode='reflect');
            gx, gy=np.gradient(in_sub.z[0,:,:])
            in_sub.z[0,:,:]=gx/dx;
            # anywhere the gradient of the mask is nonzero, set the mask to zero
            mask=np.float32(mask)
            [gxm, gym]=np.gradient(mask[0,:,:])      
            mask=np.logical_or(np.logical_or(mask, gxm!=0), gym!=0)       
            in_sub.z[mask]=np.NaN;
            if args.isotropic:
                gy_sub=im_subset(in_sub.c0, in_sub.r0, in_sub.Nc, in_sub.Nr, None,  pad_val=0, Bands=[1], stride=blocksize-2*N, pad=N, no_edges=False)
                gy_sub.z=np.array(gy)/dx;
                gy_sub.z.shape=in_sub.z.shape;
                gy_sub.z[mask]=np.NaN;
            
        if args.take_log:
           in_sub.z[(in_sub.z==0) | (in_sub.z==inNoData)] = np.NaN
           in_sub.z=np.log10(in_sub.z)
           
        parallelInputList=list();
        for fft_sub in im_subset(in_sub.c0, in_sub.r0, blocksize, blocksize, in_sub, Bands=[[1]], stride=dec, pad=(N-dec)/2, no_edges=True):
            if fft_sub.z.dtype == 'int8':
                fft_sub.z=fft_sub.z.view(np.uint8);
            if args.take_slope:
                mask=np.isnan(fft_sub.z)
            else:
                mask=np.logical_or(np.isnan(fft_sub.z), fft_sub.z==inNoData)
            if np.any(mask):
                continue

            if args.isotropic:
                fft_sub_y=im_subset(fft_sub.c0, fft_sub.r0, blocksize, blocksize, gy_sub, Bands=[[1]], stride=dec, pad=(N-dec)/2, no_edges=True);
                fft_sub_y.setBounds(fft_sub.c0, fft_sub.r0, fft_sub.Nc, fft_sub.Nr, update=True)
                imageData=np.zeros([2, fft_sub.Nc, fft_sub.Nr])
                imageData[0,:,:]=np.float64(fft_sub.z[0,:,:])
                imageData[1,:,:]=np.float64(fft_sub_y.z[0,:,:])
            else:
                imageData=np.float64(fft_sub.z[0,:,:])
            r_out=int((fft_sub.r0+N/2)/dec)-P_sub.r0
            c_out=int((fft_sub.c0+N/2)/dec)-P_sub.c0
            dd=np.float64(fft_sub.z[0,:,:])
            if np.sum(dd.ravel)==0:
                continue
            if args.num_processes==1:
                 P_i, az_i, R_i, bar_i, fft_time_i=get_P(imageData, W, L_bins, kx.ravel(), ky.ravel(), use_fftw, Wsum=Wsum, Wsum2=Wsum2, use_mean=args.use_mean)
                 P_sub.z[:, r_out, c_out]=np.log10( P_i.ravel()/N**4.)
                 az_sub.z[:,r_out, c_out]=az_i.ravel()
                 R_sub.z[:, r_out, c_out]=np.log10(R_i.ravel())
                 if bar_i == 0:
                     Ps_sub.z[:, r_out, c_out]=np.NaN;
                 else:
                    Ps_sub.z[:, r_out, c_out]=np.log10( P_i.ravel()/N**4.)-np.log10(np.abs(bar_i)**2)
                 total_fft_time=total_fft_time+fft_time_i
            else:
                #P_i, az_i, R_i, bar_i=get_P(W, imageData, L_bins, kx.ravel(), ky.ravel(), use_fftw, Wsum=Wsum, Wsum2=Wsum2, use_mean=args.use_mean)
                parallelInputList.append( (imageData, W, L_bins, kx.ravel(), ky.ravel(), use_fftw, Wsum, Wsum2, args.use_mean, r_out, c_out))
        if args.num_processes>1:
            # now run the jobs in parallel, get their output (in random order)
            parallelOutputList=myPool.map(get_P_wrapper, parallelInputList)                              
            #parallelOutputList=[get_P_wrapper(thing) for thing in parallelInputList]
            for parallelItem in parallelOutputList:
                r_out, c_out, P_i, az_i, R_i, bar_i, fft_time_i = parallelItem
                P_sub.z[:, r_out, c_out]=np.log10( P_i.ravel()/N**4.)
                az_sub.z[:,r_out, c_out]=az_i.ravel()
                R_sub.z[:, r_out, c_out]=np.log10(R_i.ravel())
                if bar_i == 0:
                    Ps_sub.z[:, r_out, c_out]=np.NaN;
                else:
                    Ps_sub.z[:, r_out, c_out]=np.log10( P_i.ravel()/N**4.)-np.log10(np.abs(bar_i)**2)
                total_fft_time=total_fft_time+fft_time_i
        az_sub.z[az_sub.z<0]+=180. 
        az_sub.z[P_sub.z==out_nodata]=out_nodata;
        P_sub.writeSubsetTo(bands, P_sub)
        Ps_sub.writeSubsetTo(bands, Ps_sub)
        az_sub.writeSubsetTo(bands, az_sub)
        R_sub.writeSubsetTo(bands, R_sub)
        dtime=time.time()-start
        start=time.time()

    total_time=time.time()-start_time
    print("finished in %3.2f total / %3.2f of fft time" % (total_time, total_fft_time))
    # test: moving the pixels by -dec/2 
    xform[3]=xform[3]-xform[5]*dec/2
    xform[0]=xform[0]-xform[1]*dec/2
    xform[5]=xform[5]*dec
    xform[1]=xform[1]*dec

    P_Ds.SetGeoTransform(tuple(xform))
    P_Ds.SetProjection(ds.GetProjection())
    az_Ds.SetGeoTransform(tuple(xform))
    az_Ds.SetProjection(ds.GetProjection())
    R_Ds.SetGeoTransform(tuple(xform))
    R_Ds.SetProjection(ds.GetProjection())
    Ps_Ds.SetGeoTransform(tuple(xform))
    Ps_Ds.SetProjection(ds.GetProjection())
    del P_Ds
    del az_Ds
    del R_Ds
    del Ps_Ds

     
if __name__=="__main__":
    freeze_support()
    main()
