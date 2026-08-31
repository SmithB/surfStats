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
import scipy.ndimage as snd
import numpy as np
from surfStats import im_subset
import surfStats.scaleMap as sm

# pyfftw is optional; get_power picks the backend, we only decide whether to
# ask for it.  Note that the FFT entry points live in pyfftw.interfaces, not on
# the pyfftw top level.
try:
   import pyfftw
   import pyfftw.interfaces
except ImportError:
    pyfftw = None
import sys, os
import time
np.seterr(invalid='ignore')

def get_P_wrapper(argList):
    [ img, W, lambda_els, kx, ky, use_fftw, Wsum, Wsum2, use_mean, r_out, c_out]=argList
    P, az, R, bar, fft_time=sm.get_power(img, W, lambda_els, kx, ky, use_fftw=use_fftw, Wsum=Wsum, Wsum2=Wsum2, use_mean=use_mean)
    return (r_out, c_out, P, az, R, bar, fft_time)

def pgc_companion_path(input_file, suffix):
    # PGC companion products (matchtag/bitmask) are always plain GeoTIFFs,
    # regardless of what format the main DEM is stored in (.tif, .vrt, ...).
    base = os.path.splitext(input_file)[0]
    if base.endswith('_dem'):
        base = base[:-len('_dem')]
    return f"{base}_{suffix}.tif"

def _open_or_die(path, description):
    ds = gdal.Open(path)
    if ds is None:
        raise SystemExit(f"could not open {description} '{path}' with GDAL "
                          f"(check the path and that GDAL recognizes the format)")
    return ds

def mask_pgc( this_bounds, mask, pgc_subs, dec):
    for key, sub in pgc_subs.items():
        sub.setBounds(*this_bounds, update=True)
    mask *= np.squeeze((pgc_subs['bitmask'].z == 0) | (pgc_subs['bitmask'].z == 2))
    if dec > 1:
        # skip erosion if the mask is all valid
        if not np.all(pgc_subs['matchtag'].z):
            mask *= snd.binary_erosion(
                snd.binary_erosion(np.squeeze(pgc_subs['matchtag'].z), np.ones((1, dec), dtype=bool), border_value=1),
                np.ones((dec,1), dtype=bool), border_value=1)
    else:
        mask &= np.squeeze(pgc_subs['matchtag'].z).astype(bool)

def main():
    fold=True
    parser = argparse.ArgumentParser(description='calculate scale map on the specified file.  Positional arguments give the input file and the maximum scale')
    parser.add_argument('input_file', help='input DEM raster (any GDAL-readable format, e.g. GeoTIFF or VRT)')
    parser.add_argument('N_LW')
    parser.add_argument('--erode_scale', '-e', type=float, default=None)
    parser.add_argument('--landsat', action='store_true', default=False, help='apply Landsat DN-to-reflectance scaling (z*2e-5-0.1) before eroding')
    parser.add_argument('--take_slope', '-t', action='store_true', default=False)
    parser.add_argument('--take_log', '-L', action='store_true', default=False)
    parser.add_argument('--use_mean','-m', action='store_true');
    parser.add_argument('--prefilter_width','-p', type=float, default=None)
    parser.add_argument('--num_processes','-n', type=int, default=1)
    parser.add_argument('--mask_file', type=str)
    parser.add_argument('--pgc_masks', action='store_true', help='Use the PGC bitmask and matchtags to mask the input') 
    parser.add_argument('--mask_values', type=float, nargs='+')
    parser.add_argument('--out_label', type=str, help='add this string to the output file name')
    parser.add_argument('--avg_file', type=str, help='file whose values will be averaged at the same resolution as the ouput')
    parser.add_argument('--avg_name', type=str, default='aux', help='name for the quantity derived from the average file')
    parser.add_argument('--isotropic', '-i', action='store_true');
    args=parser.parse_args()

    if args.isotropic and not args.take_slope:
        parser.error("--isotropic requires --take_slope: the isotropic power "
                     "combines the x and y slope components")
    if args.mask_file is not None and not args.mask_values:
        parser.error("--mask_file requires --mask_values: without it every "
                     "pixel would be masked out")

    N_LW=int(args.N_LW)
    
    # check if pyfftw is available, activate the plan cache if it is
    use_fftw=False
    if pyfftw is not None:
        try:
            pyfftw.interfaces.cache.enable()
            use_fftw=True
        except AttributeError:
            use_fftw=False

    if args.num_processes>1:
        # setup the multiprocessing pool
        myPool=Pool(args.num_processes);
    
    out_keys=['P','Ps','az','R']

    ds=_open_or_die(args.input_file, "input file")
    if args.avg_file is not None:
        avg_ds=_open_or_die(args.avg_file, "avg file")
        avg_sub=im_subset(0, 0, 0, 0, avg_ds, pad_val=0, Bands=[1])

    if args.mask_file is not None:
        mask_ds=_open_or_die(args.mask_file, "mask file")
        mask_sub=im_subset(0, 0, 0, 0, mask_ds, pad_val=0, Bands=[1])
    out_base=os.path.splitext(args.input_file)[0];
    
    if args.use_mean is True:
        out_base=out_base+"_mean"
        
    if args.out_label is not None:
        out_base += args.out_label
        
    out_files={}
    for key in out_keys:
        out_files[key]=out_base+f'_{key}_fft2.tif'

    out_nodata=np.nan
        
    print("working on %s, outfile is %s" % (args.input_file, out_files['P']) )
    if args.use_mean is True:
        print("----------using the mean instead of the sum")
            
    for file in out_files.values():
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

    _, L, kx, ky=sm.az_lambda(N, N, 1, fold=fold)
    L_bins=[np.ravel_multi_index(np.nonzero((L>=this) & (L < 2*this)), L.shape) for this in scales]
    W=sm.hanning2(N)
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
    subs={}
    Dsets={}
    for key, file in out_files.items():
        Dsets[key] = driver.Create(file, int(nX/dec), int(nY/dec), len(scales),\
                                   gdalconst.GDT_Float32, options = ['BigTIFF=YES'])
        # Does this need stride=1?
        subs[key]=im_subset(0, 0, int( nX/dec), int(nY/dec), Dsets[key], Bands=list(bands))

    if args.avg_file is not None:
        out_files[args.avg_name]=out_base+f'_{args.avg_name}_fft2.tif'
        if os.path.exists(out_files[args.avg_name]): 
            os.remove(out_files[args.avg_name])
        Dsets[args.avg_name] = driver.Create(out_files[args.avg_name], int(nX/dec), int(nY/dec), 1,\
                                   gdalconst.GDT_Float32, options = ['BigTIFF=YES'])
        subs[args.avg_name]=im_subset(0, 0, int( nX/dec), int(nY/dec), Dsets[args.avg_name], Bands=[1])

    
    if args.pgc_masks:
        pgc_subs={}
        for key, pad_val in zip(['matchtag','bitmask'], [1, 0]):
            companion_path = pgc_companion_path(args.input_file, key)
            sub_ds=_open_or_die(companion_path, f"PGC {key} file")
            pgc_subs[key] = im_subset(0, 0, nX, nY, sub_ds, pad_val=pad_val, Bands=[1])


    total_fft_time=0.0
    start_time=time.time()
    for out_ds in Dsets.values():
        for bandN in range(1, out_ds.RasterCount+1):
            band=out_ds.GetRasterBand(bandN)
            band.SetNoDataValue(out_nodata)
            # GTiff creates the band filled with zeros, which is indistinguishable
            # from a real result.  Pre-fill with nodata so that blocks we skip
            # read back as nodata instead of as zero.
            band.Fill(out_nodata)

    if args.erode_scale is not None:
        erode_kernel=np.ones([1, int(2*args.erode_scale)], dtype=bool)

    start=time.time()
    dtime=0
    stride=blocksize-2*N
    print("nX_out=%f, nY_out=%f" % (int(nX/dec), int(nY/dec)))
    N_out_sub=int(blocksize/dec)-1
    for in_sub in im_subset(0, 0,  nX,  nY, ds, pad_val=0, Bands=[1], stride=blocksize-2*N, pad=N, no_edges=False):
        in_bounds = [in_sub.c0, in_sub.r0, in_sub.Nc, in_sub.Nr]
        out_bounds = [int((in_sub.c0+N/2)/dec), int((in_sub.r0+N/2)/dec), N_out_sub, N_out_sub]
        for sub in subs.values():
            sub.setBounds(*out_bounds)
            sub.z = np.zeros([len(sub.Bands), int( N_out_sub), int(N_out_sub)])+out_nodata

        sys.stdout.write("\r\b c0=%d/%d, r0=%d/%d, last dt=%f  " % \
                         (int(float(in_sub.c0)/float(stride)), \
                          int(float(nX)/float(stride)), \
                          int(float(in_sub.r0)/float(stride)), \
                          int(float(nY)/float(stride)), dtime))
        sys.stdout.flush()

        if args.mask_file is not None:
            mask_sub.setBounds(*in_bounds, update=True)
            in_sub.z.ravel()[~np.isin(mask_sub.z[0].ravel(), args.mask_values)]=inNoData
        
        if args.pgc_masks:
            mask = np.ones_like(in_sub.z, dtype=bool)
            mask_pgc(in_bounds, mask, pgc_subs, int(dec))
            in_sub.z.ravel()[mask.ravel()==0] = inNoData

        if np.all(np.logical_or(in_sub.z == 0, np.logical_or(np.isnan(in_sub.z), in_sub.z==inNoData))):
            # Nothing to compute.  Consecutive output blocks overlap, so writing
            # this all-nodata buffer would erase the valid pixels the previous
            # block put in the overlap.  The bands were pre-filled with nodata.
            continue

        if args.avg_file is not None:
            avg_sub.setBounds(in_sub.c0, in_sub.r0, in_sub.Nc, in_sub.Nr, update=True)
            avg_sub_sub = im_subset(0, 0, blocksize, blocksize, avg_sub)

        if args.landsat or args.erode_scale is not None:
            # derive the nodata mask from the raw DNs, before any rescaling
            nodata_mask=np.logical_or(in_sub.z == 0., np.logical_or(np.isnan(in_sub.z), in_sub.z==inNoData))

        if args.landsat:
            in_sub.z=np.maximum(0, in_sub.z*2.e-5-0.1)
            in_sub.z[nodata_mask]=inNoData

        if args.erode_scale is not None:
            mask=nodata_mask
            if np.any(mask):
                mask[0,:,:]=snd.binary_dilation(mask[0,:,:], structure=erode_kernel)
                mask[0,:,:]=snd.binary_dilation(mask[0,:,:], structure=erode_kernel.transpose())
            if np.all(mask):
                continue

            in_sub.z[mask]=inNoData
            
        if args.take_slope:
            dx=ds.GetGeoTransform()[1]
            in_sub.z=np.float32(in_sub.z)
            if args.erode_scale is None:
                mask=in_sub.z==inNoData    
            if args.prefilter_width is not None:
                #print "prefiltering with kernel of width %f" % args.prefilter_width
                in_sub.z[0,:,:]=snd.gaussian_filter(in_sub.z[0,:,:], args.prefilter_width, mode='reflect');
            gx, gy=np.gradient(in_sub.z[0,:,:])
            in_sub.z[0,:,:]=gx/dx;
            # anywhere the gradient of the mask is nonzero, set the mask to zero
            mask=np.float32(mask)
            [gxm, gym]=np.gradient(mask[0,:,:])      
            mask=np.logical_or(np.logical_or(mask, gxm!=0), gym!=0)       
            in_sub.z[mask]=np.nan
            if args.isotropic:
                gy_sub=im_subset(in_sub.c0, in_sub.r0, in_sub.Nc, in_sub.Nr, None,  pad_val=0, Bands=[1], stride=blocksize-2*N, pad=N, no_edges=False)
                gy_sub.z=np.array(gy)/dx;
                gy_sub.z.shape=in_sub.z.shape;
                gy_sub.z[mask]=np.nan
            
        if args.take_log:
           in_sub.z[(in_sub.z==0) | (in_sub.z==inNoData)] = np.nan
           in_sub.z=np.log10(in_sub.z)
           
        parallelInputList=list();
        for fft_sub in im_subset(in_sub.c0, in_sub.r0, blocksize, blocksize, in_sub, Bands=[1], stride=dec, pad=(N-dec)/2, no_edges=True):
            if fft_sub.z.dtype == 'int8':
                fft_sub.z=fft_sub.z.view(np.uint8);
            if args.take_slope:
                mask=np.isnan(fft_sub.z)
            else:
                mask=np.logical_or(np.isnan(fft_sub.z), fft_sub.z==inNoData)
            if np.any(mask):
                continue

            zx=np.float64(fft_sub.z[0,:,:])
            if np.sum(zx.ravel())==0:
                continue

            if args.isotropic:
                fft_sub_y=im_subset(fft_sub.c0, fft_sub.r0, blocksize, blocksize, gy_sub, Bands=[1], stride=dec, pad=(N-dec)/2, no_edges=True);
                fft_sub_y.setBounds(fft_sub.c0, fft_sub.r0, fft_sub.Nc, fft_sub.Nr, update=True)
                imageData=np.zeros([2, fft_sub.Nr, fft_sub.Nc])
                imageData[0,:,:]=zx
                imageData[1,:,:]=np.float64(fft_sub_y.z[0,:,:])
            else:
                imageData=zx
            r_out=int((fft_sub.r0+N/2)/dec)-subs['P'].r0
            c_out=int((fft_sub.c0+N/2)/dec)-subs['P'].c0

            if args.avg_file is not None:
                avg_sub_sub.setBounds(fft_sub.c0, fft_sub.r0, fft_sub.Nc, fft_sub.Nr, update=True)
                subs[args.avg_name].z[0,r_out, c_out]=np.nanmean(avg_sub_sub.z)

            if args.num_processes==1:
                 P_i, az_i, R_i, bar_i, fft_time_i=sm.get_power(imageData, W, L_bins, kx.ravel(), ky.ravel(), use_fftw, Wsum=Wsum, Wsum2=Wsum2, use_mean=args.use_mean)
                 subs['P'].z[:, r_out, c_out]=np.log10( P_i.ravel()/N**4.)
                 subs['az'].z[:,r_out, c_out]=az_i.ravel()
                 subs['R'].z[:, r_out, c_out]=np.log10(R_i.ravel())
                 if bar_i == 0:
                     subs['Ps'].z[:, r_out, c_out]=np.nan
                 else:
                    subs['Ps'].z[:, r_out, c_out]=np.log10( P_i.ravel()/N**4.)-np.log10(np.abs(bar_i)**2)
                 total_fft_time=total_fft_time+fft_time_i
            else:
                parallelInputList.append( (imageData, W, L_bins, kx.ravel(), ky.ravel(), use_fftw, Wsum, Wsum2, args.use_mean, r_out, c_out))
        if args.num_processes>1:
            # now run the jobs in parallel, get their output (in random order)
            parallelOutputList=myPool.map(get_P_wrapper, parallelInputList)
            for parallelItem in parallelOutputList:
                r_out, c_out, P_i, az_i, R_i, bar_i, fft_time_i = parallelItem
                subs['P'].z[:, r_out, c_out]=np.log10( P_i.ravel()/N**4.)
                subs['az'].z[:,r_out, c_out]=az_i.ravel()
                subs['R'].z[:, r_out, c_out]=np.log10(R_i.ravel())
                if bar_i == 0:
                    subs['Ps'].z[:, r_out, c_out]=np.nan
                else:
                    subs['Ps'].z[:, r_out, c_out]=np.log10( P_i.ravel()/N**4.)-np.log10(np.abs(bar_i)**2)
                total_fft_time=total_fft_time+fft_time_i
        subs['az'].z[subs['az'].z<0]+=180. 
        subs['az'].z[np.isnan(subs['P'].z)]=out_nodata
        for key, sub in subs.items():
            sub.writeSubsetTo(sub.Bands, sub)#, VERBOSE=key=='P')

        dtime=time.time()-start
        start=time.time()

    total_time=time.time()-start_time
    print("finished in %3.2f total / %3.2f of fft time" % (total_time, total_fft_time))
    # test: moving the pixels by -dec/2 
    xform[3]=xform[3]-xform[5]*dec/2
    xform[0]=xform[0]-xform[1]*dec/2
    xform[5]=xform[5]*dec
    xform[1]=xform[1]*dec

    for key, DS in Dsets.items():
        print(key)
        DS.SetGeoTransform(tuple(xform))
        DS.SetProjection(ds.GetProjection())

    sub=None
    del(Dsets)
    del(subs)
    import gc
    gc.collect()

    if args.num_processes>1:
        myPool.close()
        myPool.join()
        
if __name__=="__main__":
    freeze_support()
    main()
